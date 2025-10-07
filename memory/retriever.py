"""Memory retrieval and formatting for conversation context."""

from typing import List, Dict, Any, Optional
from memory.vector_store import VectorStore
from config import MEMORY_RETRIEVAL_COUNT


class MemoryRetriever:
    """Retrieves and formats relevant memories for conversation context."""

    def __init__(self, vector_store: VectorStore):
        """Initialize the memory retriever.

        Args:
            vector_store: VectorStore instance for searching memories
        """
        self.vector_store = vector_store

    def retrieve_relevant_memories(
        self,
        query: str,
        n_results: int = MEMORY_RETRIEVAL_COUNT,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the query.

        Args:
            query: The user's query
            n_results: Number of memories to retrieve
            conversation_id: Optional conversation ID to filter by

        Returns:
            List of relevant memories
        """
        # Search for relevant memories
        memories = self.vector_store.search_memories(
            query=query,
            n_results=n_results,
            conversation_id=conversation_id
        )

        return memories

    def format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories as a string for inclusion in prompts.

        Args:
            memories: List of memory dictionaries

        Returns:
            Formatted string of memories
        """
        if not memories:
            return ""

        # Group memories by type
        facts = []
        preferences = []
        entities = []
        summaries = []

        for mem in memories:
            mem_type = mem.get("metadata", {}).get("type", "fact")
            text = mem.get("text", "")

            if mem_type == "fact":
                facts.append(text)
            elif mem_type == "preference":
                preferences.append(text)
            elif mem_type == "entity":
                entities.append(text)
            elif mem_type == "summary":
                summaries.append(text)

        # Build formatted string
        parts = []

        if facts:
            parts.append("Facts about the user:\n- " + "\n- ".join(facts[:3]))

        if preferences:
            parts.append("User preferences:\n- " + "\n- ".join(preferences[:3]))

        if summaries:
            parts.append("Previous conversation context:\n- " + "\n- ".join(summaries[:2]))

        if parts:
            return "Relevant information from previous conversations:\n\n" + "\n\n".join(parts)

        return ""

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories.

        Returns:
            Dictionary with memory statistics
        """
        total_count = self.vector_store.count_memories()

        # Get sample of memories by type
        all_memories = self.vector_store.get_all_memories(limit=100)

        type_counts = {}
        for mem in all_memories:
            mem_type = mem.get("metadata", {}).get("type", "unknown")
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        return {
            "total_memories": total_count,
            "type_breakdown": type_counts
        }

    def get_conversation_memories(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all memories for a specific conversation.

        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of memories to return

        Returns:
            List of memories
        """
        return self.vector_store.get_all_memories(
            conversation_id=conversation_id,
            limit=limit
        )
