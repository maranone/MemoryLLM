"""Vector database interface using ChromaDB for memory storage."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL


class VectorStore:
    """Manages persistent memory storage using ChromaDB."""

    def __init__(self, collection_name: str = "conversation_memories"):
        """Initialize ChromaDB client and collection.

        Args:
            collection_name: Name of the collection to use
        """
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def close(self) -> None:
        """Close the ChromaDB client connection."""
        try:
            # Clear references
            self.collection = None
            # ChromaDB doesn't have an explicit close, but we can try to clean up
            if hasattr(self.client, '_producer'):
                self.client._producer = None
            if hasattr(self.client, '_consumer'):
                self.client._consumer = None
            self.client = None
        except Exception:
            pass

    def add_memory(
        self,
        text: str,
        memory_type: str,
        conversation_id: str,
        source_context: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a memory to the vector store.

        Args:
            text: The memory content
            memory_type: Type of memory (fact, preference, summary, entity)
            conversation_id: ID of the conversation this memory came from
            source_context: Original conversation snippet
            metadata: Additional metadata

        Returns:
            ID of the stored memory
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Prepare metadata
        mem_metadata = {
            "type": memory_type,
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "source_context": source_context[:500] if source_context else "",  # Limit context length
        }

        if metadata:
            mem_metadata.update(metadata)

        # Add to collection
        self.collection.add(
            ids=[memory_id],
            documents=[text],
            metadatas=[mem_metadata]
        )

        return memory_id

    def search_memories(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories using semantic similarity.

        Args:
            query: The search query
            n_results: Number of results to return
            memory_type: Filter by memory type
            conversation_id: Filter by conversation ID

        Returns:
            List of memory dictionaries with text, metadata, and distance
        """
        # Build where filter
        where_filter = {}
        if memory_type:
            where_filter["type"] = memory_type
        if conversation_id:
            where_filter["conversation_id"] = conversation_id

        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )

        # Format results
        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                memory = {
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "id": results["ids"][0][i] if results["ids"] else None
                }
                memories.append(memory)

        return memories

    def get_all_memories(
        self,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all memories, optionally filtered by conversation.

        Args:
            conversation_id: Filter by conversation ID
            limit: Maximum number of memories to return

        Returns:
            List of memory dictionaries
        """
        where_filter = {"conversation_id": conversation_id} if conversation_id else None

        results = self.collection.get(
            where=where_filter,
            limit=limit
        )

        memories = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                memory = {
                    "text": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    "id": results["ids"][i] if results["ids"] else None
                }
                memories.append(memory)

        return memories

    def delete_memory(self, memory_id: str) -> None:
        """Delete a specific memory by ID.

        Args:
            memory_id: ID of the memory to delete
        """
        self.collection.delete(ids=[memory_id])

    def clear_conversation_memories(self, conversation_id: str) -> None:
        """Delete all memories for a specific conversation.

        Args:
            conversation_id: ID of the conversation
        """
        self.collection.delete(where={"conversation_id": conversation_id})

    def count_memories(self) -> int:
        """Get total count of stored memories.

        Returns:
            Number of memories in the store
        """
        return self.collection.count()
