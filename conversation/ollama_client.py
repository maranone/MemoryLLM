"""Ollama API client wrapper for LLM inference."""

import ollama
from typing import AsyncGenerator, List, Dict, Any, Optional
import asyncio

from config import OLLAMA_MODEL, OLLAMA_HOST


class OllamaClient:
    """Wrapper for Ollama API with streaming support."""

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        """Initialize Ollama client.

        Args:
            model: Model name to use
            host: Ollama server host
        """
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        """Generate a non-streaming response.

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens if max_tokens else -1
                }
            )

            return response["message"]["content"]

        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")

    async def generate_streaming(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt
            temperature: Sampling temperature

        Yields:
            Chunks of generated text
        """
        formatted_messages = []

        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        formatted_messages.extend(messages)

        try:
            # Run streaming in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            # Create a queue for streaming chunks
            queue = asyncio.Queue()

            def stream_worker():
                try:
                    stream = self.client.chat(
                        model=self.model,
                        messages=formatted_messages,
                        stream=True,
                        options={"temperature": temperature}
                    )

                    for chunk in stream:
                        content = chunk["message"]["content"]
                        asyncio.run_coroutine_threadsafe(
                            queue.put(content),
                            loop
                        )

                    asyncio.run_coroutine_threadsafe(
                        queue.put(None),  # Signal end of stream
                        loop
                    )

                except Exception as e:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(Exception(f"Streaming failed: {e}")),
                        loop
                    )

            # Start streaming in background thread
            loop.run_in_executor(None, stream_worker)

            # Yield chunks as they arrive
            while True:
                chunk = await queue.get()

                if chunk is None:  # End of stream
                    break

                if isinstance(chunk, Exception):
                    raise chunk

                yield chunk

        except Exception as e:
            raise Exception(f"Ollama streaming failed: {e}")

    def check_connection(self) -> bool:
        """Check if Ollama server is accessible.

        Returns:
            True if connection successful
        """
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models.

        Returns:
            List of model names
        """
        try:
            response = self.client.list()
            # Handle different response formats
            if isinstance(response, dict) and "models" in response:
                return [model.get("name", model.get("model", "")) for model in response["models"]]
            elif hasattr(response, 'models'):
                # Handle object response with models attribute
                models = []
                for model in response.models:
                    if hasattr(model, 'model'):
                        models.append(model.model)
                    elif hasattr(model, 'name'):
                        models.append(model.name)
                    else:
                        models.append(str(model))
                return models
            else:
                return []
        except Exception as e:
            raise Exception(f"Failed to list models: {e}")

    async def generate_with_context(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Generate response with full conversation context.

        Args:
            user_message: Current user message
            conversation_history: Previous conversation turns
            system_prompt: System prompt
            temperature: Sampling temperature

        Yields:
            Chunks of generated text
        """
        # Build messages list
        messages = []

        # Add conversation history
        messages.extend(conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Stream response
        async for chunk in self.generate_streaming(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature
        ):
            yield chunk
