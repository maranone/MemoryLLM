"""Main entry point for LLM Memory System with Ollama."""

import asyncio
import sys
import uuid
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner

from conversation.ollama_client import OllamaClient
from conversation.context_manager import ContextManager
from memory.vector_store import VectorStore
from memory.extractor import MemoryExtractor
from memory.retriever import MemoryRetriever
from config import (
    OLLAMA_MODEL,
    OLLAMA_HOST,
    SYSTEM_PROMPT,
    CONTEXT_WINDOW_SIZE,
    EXTRACTION_FREQUENCY
)
import json
import os
from pathlib import Path


class MemoryChatSystem:
    """Main chat system with memory capabilities."""

    def __init__(self):
        """Initialize the chat system."""
        self.console = Console()
        self.conversation_id = str(uuid.uuid4())

        # Initialize components
        self.ollama_client = OllamaClient(model=OLLAMA_MODEL, host=OLLAMA_HOST)
        self.context_manager = ContextManager(max_turns=CONTEXT_WINDOW_SIZE)
        self.vector_store = VectorStore()
        self.memory_extractor = MemoryExtractor(self.ollama_client)
        self.memory_retriever = MemoryRetriever(self.vector_store)

        self.turn_count = 0
        self.memory_update_tasks = []
        self.current_session_file = None  # Track current session file for auto-save
        self.user_name = None  # Will be loaded in initialize
        self.assistant_name = None  # Will be loaded in initialize
        self.debug_mode = False  # Debug mode flag

    def _load_user_config(self) -> tuple[Optional[str], Optional[str]]:
        """Load user and assistant names from config file."""
        user_config_path = Path("./user_config.json")
        if user_config_path.exists():
            try:
                with open(user_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get("user_name"), config.get("assistant_name")
            except Exception:
                return None, None
        return None, None

    def _save_user_config(self, user_name: str, assistant_name: str) -> None:
        """Save user and assistant names to config file."""
        user_config_path = Path("./user_config.json")
        try:
            with open(user_config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "user_name": user_name,
                    "assistant_name": assistant_name
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save configuration: {e}")

    def _prompt_for_setup(self) -> tuple[str, str]:
        """Prompt user for setup on first run."""
        self.console.print("\n[bold cyan]👋 Welcome! It looks like this is your first time here.[/bold cyan]\n")

        # Get user name
        while True:
            user_name = self.console.input("[cyan]What's your name?[/cyan] ").strip()
            if user_name:
                break
            else:
                self.console.print("[yellow]Please enter your name.[/yellow]")

        # Get assistant name
        self.console.print(f"\n[green]Nice to meet you, {user_name}![/green]")
        self.console.print("[cyan]Now, what would you like to call your AI companion?[/cyan]")
        self.console.print("[dim](Default: Leo)[/dim]\n")

        assistant_name = self.console.input("[cyan]Assistant name:[/cyan] ").strip()
        if not assistant_name:
            assistant_name = "Leo"

        self._save_user_config(user_name, assistant_name)
        self.console.print(f"\n[green]Perfect! {assistant_name} is ready to chat with you. 🎉[/green]\n")

        return user_name, assistant_name

    async def initialize(self) -> bool:
        """Initialize and check connections.

        Returns:
            True if initialization successful
        """
        try:
            # Check Ollama connection
            if not self.ollama_client.check_connection():
                self.console.print("[red]Error: Cannot connect to Ollama server[/red]")
                self.console.print(f"[yellow]Make sure Ollama is running at {OLLAMA_HOST}[/yellow]")
                return False

            # Check if model is available
            models = self.ollama_client.list_models()
            if OLLAMA_MODEL not in models:
                self.console.print(f"[red]Error: Model '{OLLAMA_MODEL}' not found[/red]")
                self.console.print(f"[yellow]Available models: {', '.join(models)}[/yellow]")
                self.console.print(f"[yellow]Pull the model with: ollama pull {OLLAMA_MODEL}[/yellow]")
                return False

            # Load or prompt for user and assistant names
            self.user_name, self.assistant_name = self._load_user_config()
            if not self.user_name or not self.assistant_name:
                self.user_name, self.assistant_name = self._prompt_for_setup()
                # Store names as initial memories
                self.vector_store.add_memory(
                    text=f"User's name is {self.user_name}",
                    memory_type="fact",
                    conversation_id=self.conversation_id,
                    source_context="Initial setup"
                )
                self.vector_store.add_memory(
                    text=f"Assistant's name is {self.assistant_name}",
                    memory_type="fact",
                    conversation_id=self.conversation_id,
                    source_context="Initial setup"
                )

            return True

        except Exception as e:
            self.console.print(f"[red]Initialization error: {e}[/red]")
            return False

    async def process_user_message(self, user_message: str) -> str:
        """Process user message and generate response.

        Args:
            user_message: User's input message

        Returns:
            Assistant's response
        """
        # Analyze user's current communication style
        user_style = await self.memory_extractor.analyze_communication_style(user_message)

        # Retrieve relevant memories
        memories = self.memory_retriever.retrieve_relevant_memories(user_message)
        memory_context = self.memory_retriever.format_memories_for_prompt(memories)

        # Add current date/time context
        from datetime import datetime
        now = datetime.now()
        datetime_context = f"\n\nCurrent date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

        # Add communication style awareness
        style_context = f"\nUser's current communication style: {user_style}\nMatch their energy and formality level naturally. Don't rigidly copy their style, but be aware and adapt your tone accordingly."

        # Calculate time since last conversation if there are memories
        if memories:
            try:
                # Get most recent memory timestamp
                latest_timestamp = None
                for mem in memories:
                    ts = mem.get("metadata", {}).get("timestamp")
                    if ts:
                        if latest_timestamp is None or ts > latest_timestamp:
                            latest_timestamp = ts

                if latest_timestamp:
                    from datetime import datetime as dt
                    last_time = dt.fromisoformat(latest_timestamp)
                    time_diff = now - last_time

                    if time_diff.days > 0:
                        datetime_context += f"\nLast conversation was {time_diff.days} day(s) ago."
                    elif time_diff.seconds > 3600:
                        hours = time_diff.seconds // 3600
                        datetime_context += f"\nLast conversation was {hours} hour(s) ago."
                    elif time_diff.seconds > 60:
                        minutes = time_diff.seconds // 60
                        datetime_context += f"\nLast conversation was {minutes} minute(s) ago."
            except Exception:
                pass  # Silently skip if timestamp parsing fails

        # Build system prompt with custom name, memories, datetime, and style
        # Add the assistant's name at the beginning
        system_prompt_with_memory = f"You are {self.assistant_name}. " + SYSTEM_PROMPT
        system_prompt_with_memory += datetime_context
        system_prompt_with_memory += style_context
        if memory_context:
            system_prompt_with_memory += f"\n\n{memory_context}"

        # Get conversation context
        context_messages = self.context_manager.get_context_messages()

        # Generate response with streaming
        full_response = ""

        # Display assistant prefix with custom name
        self.console.print(f"[bold cyan]{self.assistant_name}:[/bold cyan] ", end="")

        try:
            async for chunk in self.ollama_client.generate_with_context(
                user_message=user_message,
                conversation_history=context_messages,
                system_prompt=system_prompt_with_memory
            ):
                full_response += chunk
                # Print chunk immediately for natural streaming feel
                self.console.print(chunk, end="", style="white")

            # New line after complete response
            self.console.print()

        except Exception as e:
            self.console.print(f"\n[red]Error generating response: {e}[/red]")
            return "I apologize, but I encountered an error generating a response."

        # Add turn to context
        self.context_manager.add_turn(user_message, full_response)
        self.turn_count += 1

        # Trigger async memory extraction
        if self.turn_count % EXTRACTION_FREQUENCY == 0:
            task = asyncio.create_task(
                self._extract_and_store_memories(user_message, full_response)
            )
            self.memory_update_tasks.append(task)

        # Auto-save session in background
        save_task = asyncio.create_task(self._auto_save_session())
        self.memory_update_tasks.append(save_task)

        # Wait for background tasks to complete silently
        await asyncio.gather(*self.memory_update_tasks, return_exceptions=True)
        self.memory_update_tasks.clear()

        return full_response

    async def _extract_and_store_memories(self, user_message: str, assistant_response: str) -> None:
        """Extract and store memories asynchronously.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
        """
        try:
            # Get recent context for extraction
            context = self.context_manager.get_summary_for_extraction(n_turns=3)

            # Extract memories
            memories = await self.memory_extractor.extract_memories(
                user_message=user_message,
                assistant_response=assistant_response,
                conversation_context=context
            )

            # Debug output
            if self.debug_mode and memories:
                self.console.print(f"\n[dim cyan]🔍 Debug - Extracted {len(memories)} memories:[/dim cyan]")
                for mem in memories:
                    self.console.print(f"[dim]  • [{mem['type']}] {mem['content']}[/dim]")

            # Store extracted memories
            for memory in memories:
                self.vector_store.add_memory(
                    text=memory["content"],
                    memory_type=memory["type"],
                    conversation_id=self.conversation_id,
                    source_context=f"User: {user_message}\nAssistant: {assistant_response[:200]}",
                    metadata={"extraction_method": memory.get("extraction_method", "unknown")}
                )

            # Debug output for no memories
            if self.debug_mode and not memories:
                self.console.print(f"\n[dim cyan]🔍 Debug - No memories extracted from this exchange[/dim cyan]")

        except Exception as e:
            # Silent failure - don't interrupt conversation
            print(f"Warning: Memory extraction failed: {e}")

    def save_session(self, filename: Optional[str] = None) -> str:
        """Save current session to file.

        Args:
            filename: Optional filename

        Returns:
            Path to saved file
        """
        # Use existing session file or create new one
        if filename is None:
            if self.current_session_file:
                filepath = Path(self.current_session_file)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"session_{timestamp}.json"
                sessions_dir = Path("./sessions")
                sessions_dir.mkdir(exist_ok=True)
                filepath = sessions_dir / filename
                self.current_session_file = str(filepath)
        else:
            sessions_dir = Path("./sessions")
            sessions_dir.mkdir(exist_ok=True)
            filepath = sessions_dir / filename
            self.current_session_file = str(filepath)

        session_data = {
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "model": OLLAMA_MODEL,
            "history": self.context_manager.get_full_history(),
            "turn_count": self.turn_count
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    async def _auto_save_session(self) -> None:
        """Auto-save session in background without blocking."""
        try:
            await asyncio.to_thread(self.save_session)
        except Exception as e:
            # Silent failure - don't interrupt conversation
            print(f"Warning: Auto-save failed: {e}")

    def load_session(self, filepath: str) -> bool:
        """Load session from file.

        Args:
            filepath: Path to session file

        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            self.conversation_id = session_data.get("conversation_id", str(uuid.uuid4()))
            self.turn_count = session_data.get("turn_count", 0)

            history = session_data.get("history", [])
            self.context_manager.load_history(history)

            # Set current session file for auto-save
            self.current_session_file = filepath

            return True

        except Exception as e:
            self.console.print(f"[red]Error loading session: {e}[/red]")
            return False

    def list_sessions(self) -> list:
        """List available session files.

        Returns:
            List of session file paths
        """
        sessions_dir = Path("./sessions")
        if not sessions_dir.exists():
            return []

        return sorted(sessions_dir.glob("session_*.json"), reverse=True)

    def show_memories(self) -> None:
        """Display stored memories."""
        stats = self.memory_retriever.get_memory_stats()

        self.console.print(Panel(
            f"[cyan]Total Memories:[/cyan] {stats['total_memories']}\n"
            f"[cyan]By Type:[/cyan] {stats['type_breakdown']}",
            title="Memory Statistics"
        ))

        # Show recent memories
        memories = self.vector_store.get_all_memories(limit=10)
        if memories:
            self.console.print("\n[cyan]Recent Memories:[/cyan]")
            for i, mem in enumerate(memories[:10], 1):
                mem_type = mem.get("metadata", {}).get("type", "unknown")
                text = mem.get("text", "")
                self.console.print(f"{i}. [{mem_type}] {text}")

    async def generate_conversation_starter(self) -> None:
        """Generate and stream a personalized conversation starter based on memories."""
        # Get all memories to understand user context
        all_memories = self.vector_store.get_all_memories(limit=20)

        # Filter out the initial setup memories (just created)
        meaningful_memories = [
            mem for mem in all_memories
            if mem.get("metadata", {}).get("source_context") != "Initial setup"
        ]

        if meaningful_memories:
            # Returning user - personalized greeting
            memory_context = "Things I remember about the user:\n"
            for mem in meaningful_memories[:10]:
                mem_type = mem.get("metadata", {}).get("type", "")
                text = mem.get("text", "")
                if text:
                    memory_context += f"- {text}\n"

            prompt = f"""Based on what you know about the user, start a conversation with a warm, personalized greeting and an engaging question related to their life or previous topics we've discussed.

{memory_context}

Keep it brief (1-2 sentences) and natural. Make them feel remembered and valued."""
        else:
            # First time user - warm introductory greeting
            prompt = f"""This is your first conversation with {self.user_name}. Start with a warm, welcoming greeting that introduces yourself and asks an open-ended question to get to know them. Keep it brief (1-2 sentences) and friendly. Do NOT say "it's good to connect again" or anything implying you've talked before."""

        # Stream the starter message with custom name
        self.console.print(f"\n[bold cyan]{self.assistant_name}:[/bold cyan] ", end="")

        try:
            messages = [{"role": "user", "content": prompt}]
            async for chunk in self.ollama_client.generate_streaming(
                messages=messages,
                system_prompt=SYSTEM_PROMPT
            ):
                self.console.print(chunk, end="", style="white")

            self.console.print("\n")
        except Exception:
            self.console.print("Hi there! How are you doing today?\n")

    async def run_chat_loop(self) -> None:
        """Run the main chat loop."""
        self.console.print(Panel(
            "[bold cyan]LLM Memory System with Ollama[/bold cyan]\n\n"
            f"Model: [yellow]{OLLAMA_MODEL}[/yellow]\n"
            f"Context Window: [yellow]{CONTEXT_WINDOW_SIZE} turns[/yellow]\n\n"
            "[dim]Commands: /new /continue /memories /model /debug /save /clear /wipe /exit[/dim]",
            title="Welcome"
        ))

        # Generate and show Leo's conversation starter (streams in real-time)
        await self.generate_conversation_starter()

        while True:
            try:
                # Get user input with personalized prompt
                user_input = self.console.input(f"\n[bold green]{self.user_name}:[/bold green] ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                    continue

                # Process message
                response = await self.process_user_message(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    async def _handle_command(self, command: str) -> None:
        """Handle special commands.

        Args:
            command: Command string
        """
        cmd = command.lower().split()[0]

        if cmd == "/exit" or cmd == "/quit":
            # Everything should already be saved since we wait after each message
            # But just in case, check for any pending tasks
            if self.memory_update_tasks:
                await asyncio.gather(*self.memory_update_tasks, return_exceptions=True)

            self.console.print("[cyan]Goodbye![/cyan]")
            sys.exit(0)

        elif cmd == "/new":
            self.conversation_id = str(uuid.uuid4())
            self.context_manager.clear_all()
            self.turn_count = 0
            self.console.print("[green]Started new conversation[/green]")

        elif cmd == "/continue":
            sessions = self.list_sessions()
            if not sessions:
                self.console.print("[yellow]No saved sessions found[/yellow]")
                return

            self.console.print("[cyan]Available sessions:[/cyan]")
            for i, session in enumerate(sessions[:10], 1):
                self.console.print(f"{i}. {session.name}")

            choice = self.console.input("Select session number: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    if self.load_session(str(sessions[idx])):
                        self.console.print("[green]Session loaded successfully[/green]")
            except ValueError:
                self.console.print("[red]Invalid selection[/red]")

        elif cmd == "/memories":
            self.show_memories()

        elif cmd == "/save":
            filepath = self.save_session()
            self.console.print(f"[green]Session saved to {filepath}[/green]")

        elif cmd == "/clear":
            self.context_manager.clear_context()
            self.console.print("[green]Context cleared (memories preserved)[/green]")

        elif cmd == "/model":
            try:
                models = self.ollama_client.list_models()
                if not models:
                    self.console.print("[yellow]No models found[/yellow]")
                    return

                self.console.print("\n[cyan]Available models:[/cyan]")
                for i, model in enumerate(models, 1):
                    current = " [green](current)[/green]" if model == self.ollama_client.model else ""
                    self.console.print(f"{i}. {model}{current}")

                choice = self.console.input("\nSelect model number (or press Enter to cancel): ").strip()
                if choice:
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(models):
                            selected_model = models[idx]
                            self.ollama_client.model = selected_model
                            self.console.print(f"[green]Switched to {selected_model}[/green]")
                        else:
                            self.console.print("[red]Invalid selection[/red]")
                    except ValueError:
                        self.console.print("[red]Invalid input[/red]")
            except Exception as e:
                self.console.print(f"[red]Error listing models: {e}[/red]")

        elif cmd == "/wipe" or cmd == "/delete":
            # Confirm before wiping everything
            confirm = self.console.input(
                "[yellow]⚠️  This will delete ALL memories, sessions, and data. Type 'YES' to confirm: [/yellow]"
            ).strip()

            if confirm == "YES":
                import shutil
                import time

                # Close ChromaDB connection first to release file locks
                self.console.print("[yellow]Closing database connections...[/yellow]")
                try:
                    # Properly close ChromaDB
                    if hasattr(self, 'vector_store'):
                        self.vector_store.close()

                    # Delete references
                    if hasattr(self, 'memory_retriever'):
                        del self.memory_retriever
                    if hasattr(self, 'memory_extractor'):
                        del self.memory_extractor
                    if hasattr(self, 'vector_store'):
                        del self.vector_store

                    # Force garbage collection
                    import gc
                    gc.collect()
                    time.sleep(1.0)  # Give time for file handles to release
                except Exception as e:
                    print(f"Warning during cleanup: {e}")

                # Try to delete ChromaDB with retries
                chroma_deleted = False
                if os.path.exists("./chroma_db"):
                    for attempt in range(3):
                        try:
                            if attempt > 0:
                                self.console.print(f"[yellow]Retry attempt {attempt + 1}/3...[/yellow]")
                                import gc
                                gc.collect()
                                time.sleep(1.5)

                            shutil.rmtree("./chroma_db")
                            self.console.print("[green]✓ Deleted all memories (ChromaDB)[/green]")
                            chroma_deleted = True
                            break
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                self.console.print(f"[red]✗ Could not delete ChromaDB: {e}[/red]")
                else:
                    chroma_deleted = True

                # Delete sessions
                sessions_deleted = False
                try:
                    if os.path.exists("./sessions"):
                        shutil.rmtree("./sessions")
                        self.console.print("[green]✓ Deleted all sessions[/green]")
                    sessions_deleted = True
                except Exception as e:
                    self.console.print(f"[red]✗ Could not delete sessions: {e}[/red]")

                # Delete user config
                config_deleted = False
                try:
                    if os.path.exists("./user_config.json"):
                        os.remove("./user_config.json")
                        self.console.print("[green]✓ Deleted user configuration[/green]")
                    config_deleted = True
                except Exception as e:
                    self.console.print(f"[red]✗ Could not delete config: {e}[/red]")

                # Summary
                if chroma_deleted and sessions_deleted and config_deleted:
                    self.console.print("\n[cyan]All data wiped successfully. Starting fresh on next run.[/cyan]")
                    self.console.print("[cyan]Goodbye![/cyan]")
                    sys.exit(0)
                else:
                    self.console.print("\n[yellow]Some files could not be deleted while the program is running.[/yellow]")
                    self.console.print("[yellow]Please close the program and manually delete these folders:[/yellow]")
                    if not chroma_deleted:
                        self.console.print("  - chroma_db/")
                    if not sessions_deleted:
                        self.console.print("  - sessions/")
                    if not config_deleted:
                        self.console.print("  - user_config.json")
            else:
                self.console.print("[yellow]Wipe cancelled[/yellow]")

        elif cmd == "/debug":
            self.debug_mode = not self.debug_mode
            status = "enabled" if self.debug_mode else "disabled"
            self.console.print(f"[cyan]Debug mode {status}[/cyan]")
            if self.debug_mode:
                self.console.print("[dim]You'll now see what memories are being extracted after each message[/dim]")

        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")


async def main():
    """Main entry point."""
    chat_system = MemoryChatSystem()

    # Initialize system
    if not await chat_system.initialize():
        return

    # Run chat loop
    await chat_system.run_chat_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
