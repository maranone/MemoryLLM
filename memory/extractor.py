"""Memory extraction from conversations using Ollama."""

import re
from typing import List, Dict, Any
import asyncio
from datetime import datetime

from conversation.ollama_client import OllamaClient
from config import OLLAMA_MODEL


class MemoryExtractor:
    """Extracts facts, preferences, and entities from conversations."""

    def __init__(self, ollama_client: OllamaClient):
        """Initialize the memory extractor.

        Args:
            ollama_client: OllamaClient instance for LLM-based extraction
        """
        self.ollama_client = ollama_client

    async def analyze_communication_style(self, user_message: str) -> str:
        """Analyze the user's current communication style/tone.

        Args:
            user_message: User's message to analyze

        Returns:
            Description of communication style
        """
        analysis_prompt = f"""Analyze the communication style of this message. Focus on:
- Formality level (very formal, formal, casual, very casual/slang)
- Emotional tone (energetic, calm, serious, playful, moody, stressed)
- Message length preference (brief/direct or detailed/chatty)

User message: "{user_message}"

Respond with a brief description (1 sentence) of their current style.
Examples:
- "Casual and brief, slightly stressed"
- "Very casual with slang, playful and energetic"
- "Formal and detailed, calm professional tone"
- "Direct and moody, wants brief responses"

Your analysis:"""

        try:
            response = await asyncio.wait_for(
                self.ollama_client.generate(analysis_prompt, max_tokens=50),
                timeout=5.0
            )
            return response.strip()
        except (asyncio.TimeoutError, Exception):
            # Default if analysis fails
            return "Casual and conversational"

    async def extract_memories(
        self,
        user_message: str,
        assistant_response: str,
        conversation_context: str = ""
    ) -> List[Dict[str, Any]]:
        """Extract memories from a conversation turn.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            conversation_context: Additional context from recent conversation

        Returns:
            List of extracted memories with type and content
        """
        memories = []

        # Use LLM as primary extraction method - it's smarter and more flexible
        llm_memories = await self._extract_with_llm(
            user_message,
            assistant_response,
            conversation_context
        )
        memories.extend(llm_memories)

        return memories

    def _extract_pattern_based(self, text: str) -> List[Dict[str, Any]]:
        """Extract memories using regex patterns.

        Args:
            text: Text to extract from

        Returns:
            List of extracted memories
        """
        memories = []

        # Pattern for "I am/I'm [something]"
        patterns = [
            (r"(?:I am|I'm|I am a|I'm a)\s+(.+?)(?:\.|,|$)", "fact"),
            (r"(?:My name is|I'm called|Call me)\s+([A-Z][a-z]+)", "fact"),
            (r"(?:I live in|I'm from|I'm based in)\s+([A-Z][a-z\s]+)", "fact"),
            (r"(?:I like|I love|I enjoy|I prefer)\s+(.+?)(?:\.|,|$)", "preference"),
            (r"(?:I don't like|I hate|I dislike)\s+(.+?)(?:\.|,|$)", "preference"),
            (r"(?:My favorite|My favourite)\s+(.+?)\s+(?:is|are)\s+(.+?)(?:\.|,|$)", "preference"),
        ]

        for pattern, mem_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if mem_type == "preference" and "favorite" in pattern:
                    content = f"Favorite {match.group(1)}: {match.group(2)}"
                else:
                    content = match.group(1).strip()

                memories.append({
                    "type": mem_type,
                    "content": content,
                    "extraction_method": "pattern"
                })

        return memories

    async def _extract_with_llm(
        self,
        user_message: str,
        assistant_response: str,
        context: str
    ) -> List[Dict[str, Any]]:
        """Use LLM to extract facts and preferences.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            context: Conversation context

        Returns:
            List of extracted memories
        """
        extraction_prompt = f"""You are a memory extraction assistant. Analyze this conversation and extract ONLY the important, lasting information worth remembering about the user.

Extract things like:
- Personal facts (name, age, location, occupation, relationships)
- Preferences and opinions (likes, dislikes, values, beliefs)
- Goals, dreams, challenges, or problems they're facing
- Important life events or context
- Emotional state or patterns

DO NOT extract:
- Temporary states ("user is frustrated today")
- Trivial details
- Things already obvious from context

Conversation:
User: {user_message}
Assistant: {assistant_response}

Format each memory as: TYPE: content
Where TYPE is either FACT or PREFERENCE or GOAL

If nothing important to remember, respond with "NONE".

Examples:
FACT: User's name is Eloi
FACT: User works as a software engineer in Seattle
PREFERENCE: User dislikes traffic and commuting
GOAL: User wants to learn async programming

Your response:"""

        try:
            response = await asyncio.wait_for(
                self.ollama_client.generate(extraction_prompt, max_tokens=200),
                timeout=10.0
            )

            memories = []
            if response and response.strip().upper() != "NONE":
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        # Support FACT, PREFERENCE, and GOAL types
                        parts = line.split(':', 1)
                        mem_type = parts[0].strip().lower()
                        content = parts[1].strip()

                        # Only accept valid memory types
                        if mem_type in ['fact', 'preference', 'goal'] and content:
                            memories.append({
                                "type": mem_type,
                                "content": content,
                                "extraction_method": "llm"
                            })

            return memories

        except asyncio.TimeoutError:
            # Skip LLM extraction if it takes too long
            return []
        except Exception as e:
            print(f"Warning: LLM extraction failed: {e}")
            return []

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities (simple pattern-based).

        Args:
            text: Text to extract from

        Returns:
            List of entity memories
        """
        memories = []

        # Extract capitalized names (simple heuristic)
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        names = re.findall(name_pattern, text)

        # Common words to exclude
        exclude = {
            'I', 'My', 'The', 'A', 'An', 'This', 'That', 'Please', 'Thanks',
            'Hello', 'Hi', 'Yes', 'No', 'Maybe', 'Sure', 'Okay'
        }

        for name in names:
            if name not in exclude and len(name) > 2:
                memories.append({
                    "type": "entity",
                    "content": name,
                    "extraction_method": "pattern"
                })

        return memories

    def create_conversation_summary(
        self,
        conversation_turns: List[Dict[str, str]],
        max_length: int = 200
    ) -> str:
        """Create a summary of conversation turns.

        Args:
            conversation_turns: List of conversation turns
            max_length: Maximum length of summary

        Returns:
            Summary string
        """
        # Simple extractive summary: take first and key parts
        if not conversation_turns:
            return ""

        summary_parts = []

        # Add first turn
        if len(conversation_turns) > 0:
            first = conversation_turns[0]
            summary_parts.append(f"User asked about: {first.get('user', '')[:100]}")

        # Add any questions or key statements
        for turn in conversation_turns[1:]:
            user_msg = turn.get('user', '')
            if '?' in user_msg and len(user_msg) < 100:
                summary_parts.append(user_msg[:100])

        summary = " | ".join(summary_parts)
        return summary[:max_length]
