"""Context window management for maintaining recent conversation history."""

from collections import deque
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from config import CONTEXT_WINDOW_SIZE


class ContextManager:
    """Manages sliding window of recent conversation turns."""

    def __init__(self, max_turns: int = CONTEXT_WINDOW_SIZE):
        """Initialize context manager.

        Args:
            max_turns: Maximum number of turns to keep in context
        """
        self.max_turns = max_turns
        self.context_window = deque(maxlen=max_turns)
        self.full_history = []  # Keep full history for session saving

    def add_turn(self, user_message: str, assistant_response: str) -> None:
        """Add a conversation turn to the context.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
        """
        turn = {
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        }

        # Add to sliding window
        self.context_window.append(turn)

        # Add to full history
        self.full_history.append(turn)

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get recent context formatted as messages for Ollama.

        Returns:
            List of message dictionaries
        """
        messages = []

        for turn in self.context_window:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        return messages

    def get_context_as_text(self) -> str:
        """Get recent context as formatted text.

        Returns:
            Formatted context string
        """
        if not self.context_window:
            return ""

        lines = []
        for i, turn in enumerate(self.context_window, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
            lines.append("")

        return "\n".join(lines)

    def get_recent_turns(self, n: int = 3) -> List[Dict[str, str]]:
        """Get the N most recent turns.

        Args:
            n: Number of recent turns to get

        Returns:
            List of recent turns
        """
        window_list = list(self.context_window)
        return window_list[-n:] if len(window_list) >= n else window_list

    def get_full_history(self) -> List[Dict[str, str]]:
        """Get the complete conversation history.

        Returns:
            List of all turns
        """
        return self.full_history.copy()

    def clear_context(self) -> None:
        """Clear the context window but keep full history."""
        self.context_window.clear()

    def clear_all(self) -> None:
        """Clear both context window and full history."""
        self.context_window.clear()
        self.full_history.clear()

    def get_turn_count(self) -> int:
        """Get total number of turns in the conversation.

        Returns:
            Number of turns
        """
        return len(self.full_history)

    def load_history(self, history: List[Dict[str, str]]) -> None:
        """Load conversation history from saved data.

        Args:
            history: List of conversation turns
        """
        self.full_history = history.copy()

        # Populate context window with most recent turns
        recent_turns = history[-self.max_turns:] if len(history) > self.max_turns else history
        self.context_window = deque(recent_turns, maxlen=self.max_turns)

    def export_to_dict(self) -> Dict[str, Any]:
        """Export context manager state to dictionary.

        Returns:
            Dictionary with context state
        """
        return {
            "max_turns": self.max_turns,
            "full_history": self.full_history,
            "turn_count": len(self.full_history)
        }

    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message.

        Returns:
            Last user message or None
        """
        if self.full_history:
            return self.full_history[-1].get("user")
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the most recent assistant message.

        Returns:
            Last assistant message or None
        """
        if self.full_history:
            return self.full_history[-1].get("assistant")
        return None

    def get_summary_for_extraction(self, n_turns: int = 5) -> str:
        """Get recent conversation summary for memory extraction.

        Args:
            n_turns: Number of recent turns to include

        Returns:
            Formatted summary string
        """
        recent = self.get_recent_turns(n_turns)

        if not recent:
            return ""

        lines = []
        for turn in recent:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")

        return "\n".join(lines)
