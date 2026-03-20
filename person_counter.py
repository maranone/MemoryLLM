#!/usr/bin/env python3
"""
Person Counter — screen-capture person tracker with daily CSV logging.

Each time a person enters the captured region a visit is opened.
When they leave, the visit is closed and one row is written to the CSV.

Usage:
    python person_counter.py             # with debug preview window
    python person_counter.py --headless  # background / no window
    python person_counter.py --select-region  # drag to pick region, save, exit

PyInstaller (single-file exe):
    pip install pyinstaller
    pyinstaller --onefile --collect-all cv2 --collect-all mss --name PersonCounter person_counter.py
"""

# ==============================================================
# CONFIGURATION  —  edit these to match your setup
# ==============================================================

# Screen region to capture.  Set to None for full primary monitor.
# Run with --select-region to set this interactively.
CAPTURE_REGION = {"top": 600, "left": 500, "width": 1300, "height": 900}

# Processing
SKIP_FRAMES    = 5    # run detection on 1 in every N frames (lower = more CPU)
FRAME_SLEEP_MS = 30   # ms to sleep between capture iterations

# Person detection
YOLO_INPUT_SIZE  = 320   # 320 = faster, 640 = more accurate (must be multiple of 32)
FACE_CONFIDENCE  = 0.15  # minimum detection confidence (0–1)
IOU_THRESHOLD    = 0.70  # NMS overlap threshold — higher = keep more overlapping boxes

# Tracking
MAX_DISAPPEARED   = 40   # frames of absence before a track is dropped
MAX_CENTROID_DIST = 120
MIN_CONFIRM_FRAMES = 3   # frames before a visit is opened

# Preview grid — fixed canvas, grid adapts to number of active people
PREVIEW_W = 960    # total canvas width  (pixels)
PREVIEW_H = 1200   # total canvas height (pixels)
GRID_MAX  = 36     # max slots (6×6)
DEBUG_WINDOW_NAME = "Person Counter (press Q to quit)"

# Re-identification (--clip / --osnet flags)
CLIP_SIMILARITY_THRESHOLD  = 0.80  # CLIP cosine similarity threshold
OSNET_SIMILARITY_THRESHOLD = 0.80  # OSNet cosine similarity threshold (re-ID trained, lower is fine)
REID_MAX_EMBEDDINGS        = 5     # embeddings stored per person (FIFO)

# ==============================================================
# IMPORTS
# ==============================================================

import sys
import math
import time
import csv
import uuid
import json
import queue
import threading
import argparse
import subprocess
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Callable
import urllib.request

import numpy as np
import cv2
import mss

# ==============================================================
# PATH HELPERS  (PyInstaller-aware)
# ==============================================================

_MODEL_DIR_HOME = Path.home() / ".person_counter" / "models"


def _get_model_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "models"
        if bundled.exists():
            return bundled
    return _MODEL_DIR_HOME


def _get_exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


_CONFIG_FILE = _get_exe_dir() / "person_counter_config.json"


def _load_region() -> Optional[Dict]:
    try:
        if _CONFIG_FILE.exists():
            data = json.loads(_CONFIG_FILE.read_text())
            r = data.get("capture_region")
            if r and all(k in r for k in ("top", "left", "width", "height")):
                return r
    except Exception:
        pass
    return None


def _save_region(region: Dict) -> None:
    try:
        existing = {}
        if _CONFIG_FILE.exists():
            existing = json.loads(_CONFIG_FILE.read_text())
        existing["capture_region"] = region
        _CONFIG_FILE.write_text(json.dumps(existing, indent=2))
    except Exception as exc:
        print(f"[config] Could not save config: {exc}")


def _active_region() -> Optional[Dict]:
    return _load_region() or CAPTURE_REGION


# ==============================================================
# INTERACTIVE REGION SELECTOR
# ==============================================================

def select_region_interactive() -> Optional[Dict]:
    print("[region] Taking fullscreen screenshot ...")
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        full = np.array(shot)[:, :, :3]
        mon_left, mon_top = monitor["left"], monitor["top"]
        mon_w, mon_h = monitor["width"], monitor["height"]

    scale = min(1.0, 1280 / mon_w)
    disp_w, disp_h = int(mon_w * scale), int(mon_h * scale)
    display = cv2.resize(full, (disp_w, disp_h))
    state = {"start": None, "end": None, "drawing": False, "done": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(start=(x, y), end=(x, y), drawing=True, done=False)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state.update(end=(x, y), drawing=False, done=True)

    win = "Select region - drag, then press ENTER to confirm (ESC to cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, on_mouse)
    print("[region] Drag a rectangle, press ENTER/SPACE to confirm, ESC to cancel.")

    result = None
    while True:
        canvas = display.copy()
        if state["start"] and state["end"]:
            cv2.rectangle(canvas, state["start"], state["end"], (0, 255, 0), 2)
            rx1 = min(state["start"][0], state["end"][0])
            ry1 = min(state["start"][1], state["end"][1])
            rx2 = max(state["start"][0], state["end"][0])
            ry2 = max(state["start"][1], state["end"][1])
            cv2.putText(canvas, f"{int((rx2-rx1)/scale)} x {int((ry2-ry1)/scale)} px",
                        (rx1 + 4, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and state["done"]:
            rx1 = min(state["start"][0], state["end"][0])
            ry1 = min(state["start"][1], state["end"][1])
            rx2 = max(state["start"][0], state["end"][0])
            ry2 = max(state["start"][1], state["end"][1])
            result = {
                "top":    mon_top + int(ry1 / scale),
                "left":   mon_left + int(rx1 / scale),
                "width":  max(1, int((rx2 - rx1) / scale)),
                "height": max(1, int((ry2 - ry1) / scale)),
            }
            break
        if key == 27:
            print("[region] Cancelled.")
            break
    cv2.destroyAllWindows()
    return result


# ==============================================================
# MODEL DOWNLOADER
# ==============================================================

class ModelDownloader:
    def ensure_all(self) -> None:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            print("[models] Installing ultralytics ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"],
                stdout=subprocess.DEVNULL,
            )
        pt = _MODEL_DIR_HOME / "yolov10n.pt"
        if pt.exists():
            print(f"[models] yolov10n.pt present ({pt.stat().st_size // 1024} KB).")
        else:
            print("[models] yolov10n.pt not found — will download on first inference.")


# ==============================================================
# SCREEN CAPTURE  (daemon thread)
# ==============================================================

class ScreenCapture:
    def __init__(self, region: Optional[Dict]) -> None:
        self._region = region
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="capture")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _loop(self) -> None:
        sleep_s = FRAME_SLEEP_MS / 1000.0
        with mss.mss() as sct:
            monitor = self._region if self._region else sct.monitors[1]
            while not self._stop.is_set():
                img = sct.grab(monitor)
                with self._lock:
                    self._frame = np.array(img)[:, :, :3]  # BGRA -> BGR
                time.sleep(sleep_s)


# ==============================================================
# PERSON DETECTOR  (YOLOv10n via ultralytics)
# ==============================================================

class PersonDetector:
    _PERSON_CLASS = 0

    def __init__(self) -> None:
        import os, shutil
        from ultralytics import YOLO
        pt_path = _MODEL_DIR_HOME / "yolov10n.pt"
        _MODEL_DIR_HOME.mkdir(parents=True, exist_ok=True)
        if pt_path.exists():
            self._model = YOLO(str(pt_path))
        else:
            # ultralytics downloads to CWD; redirect to model dir
            old_cwd = os.getcwd()
            os.chdir(str(_MODEL_DIR_HOME))
            try:
                self._model = YOLO("yolov10n.pt")
            finally:
                os.chdir(old_cwd)
        self._input_size = YOLO_INPUT_SIZE

    def detect(self, frame: np.ndarray) -> List[Dict]:
        results = self._model(
            frame,
            imgsz=self._input_size,
            conf=FACE_CONFIDENCE,
            iou=IOU_THRESHOLD,
            classes=[self._PERSON_CLASS],
            verbose=False,
        )[0]
        h, w = frame.shape[:2]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                detections.append({"bbox": (x1, y1, x2, y2), "confidence": float(box.conf[0])})
        return detections


# ==============================================================
# CENTROID TRACKER
# ==============================================================

@dataclass
class Track:
    track_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    disappeared: int = 0
    frame_count: int = 0
    confirmed: bool = False


class CentroidTracker:
    def __init__(
        self,
        on_confirmed: Callable[[Track], None],
        on_lost: Callable[[Track], None],
    ) -> None:
        self._on_confirmed = on_confirmed
        self._on_lost = on_lost
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0

    @property
    def tracks(self) -> Dict[int, Track]:
        return dict(self._tracks)

    def update(self, detections: List[Dict]) -> Dict[int, Track]:
        if not detections:
            for t in list(self._tracks.values()):
                t.disappeared += 1
                if t.disappeared > MAX_DISAPPEARED:
                    self._on_lost(self._tracks.pop(t.track_id))
            return self.tracks

        new_cents  = [self._centroid(d["bbox"]) for d in detections]
        new_bboxes = [d["bbox"] for d in detections]

        if not self._tracks:
            for c, b in zip(new_cents, new_bboxes):
                self._register(c, b)
            return self.tracks

        track_ids = list(self._tracks.keys())
        cost = self._cost_matrix(
            [self._tracks[tid].centroid for tid in track_ids],
            [self._tracks[tid].bbox for tid in track_ids],
            new_cents, new_bboxes,
        )
        used_rows: set = set()
        used_cols: set = set()
        for val, r, c in sorted(
            ((cost[r, c], r, c) for r in range(len(track_ids)) for c in range(len(detections))),
            key=lambda x: x[0],
        ):
            if val >= 1.5:
                break
            if r in used_rows or c in used_cols:
                continue
            t = self._tracks[track_ids[r]]
            t.centroid, t.bbox, t.disappeared = new_cents[c], new_bboxes[c], 0
            t.frame_count += 1
            if not t.confirmed and t.frame_count >= MIN_CONFIRM_FRAMES:
                t.confirmed = True
                self._on_confirmed(t)
            used_rows.add(r)
            used_cols.add(c)

        for r, tid in enumerate(track_ids):
            if r not in used_rows:
                self._tracks[tid].disappeared += 1
                if self._tracks[tid].disappeared > MAX_DISAPPEARED:
                    self._on_lost(self._tracks.pop(tid))

        for c in range(len(detections)):
            if c not in used_cols:
                self._register(new_cents[c], new_bboxes[c])

        return self.tracks

    def _register(self, centroid, bbox) -> None:
        self._tracks[self._next_id] = Track(
            track_id=self._next_id, centroid=centroid, bbox=bbox, frame_count=1,
        )
        self._next_id += 1

    @staticmethod
    def _centroid(bbox) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _cost_matrix(self, tc, tb, dc, db) -> np.ndarray:
        n, m = len(tc), len(dc)
        mat = np.full((n, m), 2.0, dtype=np.float32)
        for r in range(n):
            for c in range(m):
                dist = float(np.hypot(tc[r][0] - dc[c][0], tc[r][1] - dc[c][1]))
                if dist > MAX_CENTROID_DIST * 2.5:
                    continue
                mat[r, c] = (0.45 * min(dist / MAX_CENTROID_DIST, 1.0)
                              + 0.55 * (1.0 - self._iou(tb[r], db[c])))
        return mat

    @staticmethod
    def _iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union > 0 else 0.0


# ==============================================================
# CLIP EMBEDDER  (open-clip-torch, optional)
# ==============================================================

class CLIPEmbedder:
    def __init__(self) -> None:
        try:
            import open_clip
        except ImportError:
            print("[clip] Installing open-clip-torch ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "open-clip-torch", "--quiet"]
            )
            import open_clip
        import torch
        self._torch = torch
        print("[clip] Loading CLIP ViT-B/32 ...")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self._model.eval()

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        try:
            from PIL import Image
            pil    = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = self._preprocess(pil).unsqueeze(0)
            with self._torch.no_grad():
                feat = self._model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat[0].numpy()
        except Exception as exc:
            print(f"  [clip] embed error: {exc}")
            return None


# ==============================================================
# OSNET EMBEDDER  (torchreid, dedicated person re-ID)
# ==============================================================

class OSNetEmbedder:
    def __init__(self) -> None:
        try:
            import torchreid
        except (ImportError, ModuleNotFoundError):
            print("[osnet] Installing torchreid + gdown ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "gdown", "torchreid", "--quiet"]
            )
            import torchreid
        import torch
        import torchvision.transforms as T
        self._torch = torch
        print("[osnet] Loading OSNet x1.0 (pretrained MSMT17) ...")
        self._model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=True
        )
        self._model.eval()
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        try:
            rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0)
            with self._torch.no_grad():
                feat = self._model(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat[0].numpy()
        except Exception as exc:
            print(f"  [osnet] embed error: {exc}")
            return None


# ==============================================================
# PERSON ID STORE  (cosine similarity, works with any embedding)
# ==============================================================

class PersonIDStore:
    def __init__(self, threshold: float, max_embeddings: int = 5) -> None:
        self._lock      = threading.Lock()
        self._threshold = threshold
        self._max_emb   = max_embeddings
        self._next_id   = 1
        self._persons: Dict[int, List[np.ndarray]] = {}

    def identify(self, embedding: np.ndarray) -> Tuple[int, bool]:
        """Returns (person_id, is_new)."""
        with self._lock:
            best_id, best_sim = None, -1.0
            for pid, embs in self._persons.items():
                sim = max(float(np.dot(embedding, e)) for e in embs)
                if sim > best_sim:
                    best_sim, best_id = sim, pid
            if best_id is not None and best_sim >= self._threshold:
                embs = self._persons[best_id]
                embs.append(embedding)
                if len(embs) > self._max_emb:
                    embs.pop(0)
                return best_id, False
            pid = self._next_id
            self._next_id += 1
            self._persons[pid] = [embedding]
            return pid, True

    def reset_daily(self) -> None:
        with self._lock:
            self._persons.clear()
            self._next_id = 1


# ==============================================================
# CSV LOGGER
# ==============================================================

_CSV_FIELDS = ["visit_id", "person_id", "date", "first_seen", "last_seen", "duration_seconds"]


class CSVLogger:
    def __init__(self) -> None:
        self._out_dir = _get_exe_dir()
        self._lock = threading.Lock()

    def log_visit(self, record: Dict) -> None:
        with self._lock:
            path = self._out_dir / f"persons_{date.today()}.csv"
            write_header = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow(record)


# ==============================================================
# PERSON COUNTER  (orchestrator)
# ==============================================================

class PersonCounter:
    def __init__(self, headless: bool = False, use_clip: bool = False, use_osnet: bool = False) -> None:
        self._headless  = headless
        print("[init] Loading person detector (YOLOv10n) ...")
        self._detector = PersonDetector()
        self._logger   = CSVLogger()
        self._tracker  = CentroidTracker(
            on_confirmed=self._on_confirmed,
            on_lost=self._on_lost,
        )
        region = _active_region()
        print(f"[init] Capture region: {region or 'full primary monitor'}")
        self._capture = ScreenCapture(region)

        self._active_visits: Dict[int, datetime] = {}
        self._last_tracks:   Dict[int, Track]    = {}
        self._grid_slots:    List[Optional[int]] = [None] * GRID_MAX
        self._track_to_slot: Dict[int, int]      = {}
        self._current_date = date.today()
        self._frame_idx    = 0

        # Re-ID pipeline — osnet takes priority over clip if both passed
        if use_osnet:
            self._embedder  = OSNetEmbedder()
            self._id_store  = PersonIDStore(OSNET_SIMILARITY_THRESHOLD, REID_MAX_EMBEDDINGS)
            self._reid_mode = "osnet"
        elif use_clip:
            self._embedder  = CLIPEmbedder()
            self._id_store  = PersonIDStore(CLIP_SIMILARITY_THRESHOLD, REID_MAX_EMBEDDINGS)
            self._reid_mode = "clip"
        else:
            self._reid_mode = ""

        if self._reid_mode:
            self._reid_queue:  queue.Queue    = queue.Queue(maxsize=20)
            self._reid_data:   Dict[int, int] = {}   # track_id -> person_id
            self._reid_lock    = threading.Lock()
            self._reid_thread  = threading.Thread(
                target=self._reid_worker, daemon=True, name="reid"
            )

    def run(self) -> None:
        print(f"[counter] Started.  CSV -> {_get_exe_dir()}")
        if not self._headless:
            print("[counter] Press Q in the preview window to quit.")
        else:
            print("[counter] Headless mode. Press Ctrl+C to quit.")

        self._capture.start()
        if self._reid_mode:
            self._reid_thread.start()
        if not self._headless:
            cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DEBUG_WINDOW_NAME, PREVIEW_W, PREVIEW_H)
        sleep_s = FRAME_SLEEP_MS / 1000.0
        try:
            while True:
                self._check_daily_reset()
                frame = self._capture.get_latest_frame()
                if frame is None:
                    time.sleep(sleep_s)
                    continue

                self._frame_idx += 1
                if self._frame_idx % SKIP_FRAMES == 0:
                    detections = self._detector.detect(frame)
                    new_tracks = self._tracker.update(detections)
                    for tid in new_tracks:
                        if tid not in self._last_tracks and tid not in self._track_to_slot:
                            self._assign_grid_slot(tid)
                    self._last_tracks = new_tracks
                    if self._reid_mode:
                        with self._reid_lock:
                            missing = [tid for tid in new_tracks
                                       if tid not in self._reid_data
                                       and new_tracks[tid].confirmed]
                        for tid in missing:
                            x1, y1, x2, y2 = new_tracks[tid].bbox
                            crop = frame[y1:y2, x1:x2].copy()
                            if crop.size > 0:
                                try:
                                    self._reid_queue.put_nowait((tid, crop))
                                except queue.Full:
                                    break

                if not self._headless:
                    self._show_debug(frame, self._last_tracks)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                time.sleep(sleep_s)

        except KeyboardInterrupt:
            pass
        finally:
            self._capture.stop()
            for tid in list(self._active_visits.keys()):
                self._free_grid_slot(tid)
                self._flush_visit(tid, reason="shutdown")
            if not self._headless:
                cv2.destroyAllWindows()
            print("[counter] Stopped.")

    def _on_confirmed(self, track: Track) -> None:
        self._active_visits[track.track_id] = datetime.now()
        print(f"  [+] Person #{track.track_id} entered")

    def _on_lost(self, track: Track) -> None:
        self._free_grid_slot(track.track_id)
        self._flush_visit(track.track_id, reason="left")
        if self._reid_mode:
            with self._reid_lock:
                self._reid_data.pop(track.track_id, None)

    def _flush_visit(self, track_id: int, reason: str = "left") -> None:
        first_seen = self._active_visits.pop(track_id, None)
        if first_seen is None:
            return
        now      = datetime.now()
        duration = int((now - first_seen).total_seconds())
        if self._reid_mode:
            with self._reid_lock:
                pid = self._reid_data.get(track_id)
            person_label = f"P{pid}" if pid else ""
        else:
            person_label = ""
        self._logger.log_visit({
            "visit_id":         str(uuid.uuid4()),
            "person_id":        person_label,
            "date":             str(date.today()),
            "first_seen":       first_seen.strftime("%H:%M:%S"),
            "last_seen":        now.strftime("%H:%M:%S"),
            "duration_seconds": duration,
        })
        label = person_label or f"#{track_id}"
        print(f"  [-] {label} left after {duration}s  ({reason})")

    def _reid_worker(self) -> None:
        while True:
            try:
                track_id, crop = self._reid_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                embedding = self._embedder.embed(crop)
                if embedding is not None:
                    person_id, is_new = self._id_store.identify(embedding)
                    with self._reid_lock:
                        self._reid_data[track_id] = person_id
                    tag = "new" if is_new else "returning"
                    print(f"  [{self._reid_mode}] Track #{track_id} -> P{person_id} ({tag})")
            except Exception as exc:
                print(f"  [{self._reid_mode}] Error on track #{track_id}: {exc}")
            finally:
                self._reid_queue.task_done()

    def _assign_grid_slot(self, track_id: int) -> None:
        for i, occupant in enumerate(self._grid_slots):
            if occupant is None:
                self._grid_slots[i] = track_id
                self._track_to_slot[track_id] = i
                return

    def _free_grid_slot(self, track_id: int) -> None:
        slot = self._track_to_slot.pop(track_id, None)
        if slot is not None:
            self._grid_slots[slot] = None

    def _check_daily_reset(self) -> None:
        today = date.today()
        if today != self._current_date:
            print(f"[counter] New day ({today}) - resetting.")
            self._current_date = today
            for tid in list(self._active_visits.keys()):
                self._flush_visit(tid, reason="day-end")
            if self._reid_mode:
                self._id_store.reset_daily()
                with self._reid_lock:
                    self._reid_data.clear()

    def _show_debug(self, frame: np.ndarray, tracks: Dict[int, Track]) -> None:
        occupied = [(i, tid) for i, tid in enumerate(self._grid_slots) if tid is not None]

        n    = max(1, len(occupied))
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, math.ceil(n / cols))
        cw   = PREVIEW_W // cols
        ch   = PREVIEW_H // rows
        label_h = 22

        canvas = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)

        for pos, (slot_idx, track_id) in enumerate(occupied):
            r, c = divmod(pos, cols)
            x_off, y_off = c * cw, r * ch

            if track_id not in tracks:
                continue

            track = tracks[track_id]
            if not track.confirmed:
                continue

            x1, y1, x2, y2 = track.bbox
            shift = max(1, int((y2 - y1) * 0.10))
            y1 = max(0, y1 - shift)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            content_h = ch - label_h
            crop_h, crop_w = crop.shape[:2]
            scale = min(cw / crop_w, content_h / crop_h)
            nw, nh = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
            px = x_off + (cw - nw) // 2
            py = y_off + label_h + (content_h - nh) // 2
            canvas[py:py + nh, px:px + nw] = resized

            color = (0, 220, 0)
            cv2.rectangle(canvas, (x_off, y_off), (x_off + cw, y_off + label_h), color, -1)
            if self._reid_mode:
                with self._reid_lock:
                    pid = self._reid_data.get(track_id)
                label = f"P{pid} #{track_id}" if pid else f"#{track_id} ?"
            else:
                label = f"#{track_id}"
            cv2.putText(canvas, label, (x_off + 4, y_off + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
            cv2.rectangle(canvas, (x_off, y_off), (x_off + cw - 1, y_off + ch - 1), color, 1)

        if not occupied:
            cv2.putText(canvas, "No detections",
                        (PREVIEW_W // 2 - 80, PREVIEW_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 1)

        cv2.imshow(DEBUG_WINDOW_NAME, canvas)


# ==============================================================
# ENTRY POINT
# ==============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Person Counter - screen-capture tracker with daily CSV logging."
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run with no preview window.")
    parser.add_argument("--select-region", action="store_true",
                        help="Drag to pick capture region, save, then exit.")
    parser.add_argument("--clip", action="store_true",
                        help="Enable CLIP-based person re-identification.")
    parser.add_argument("--osnet", action="store_true",
                        help="Enable OSNet person re-identification (better than CLIP, needs torchreid).")
    args = parser.parse_args()

    print("=" * 48)
    print("  Person Counter")
    print("=" * 48)

    if args.select_region:
        region = select_region_interactive()
        if region:
            _save_region(region)
            print(f"[region] Saved: {region}")
            print(f"[region] Config: {_CONFIG_FILE}")
        return

    ModelDownloader().ensure_all()
    print("[init] Initialising ...")
    PersonCounter(headless=args.headless, use_clip=args.clip, use_osnet=args.osnet).run()


if __name__ == "__main__":
    main()
