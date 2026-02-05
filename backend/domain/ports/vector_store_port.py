"""Vector Store Port - Interface for vector database operations."""

from abc import ABC, abstractmethod
from typing import Any


class VectorStorePort(ABC):
    """Port interface for vector database operations."""

    @abstractmethod
    async def create_collection(self, name: str) -> dict:
        """Create a new collection.

        Args:
            name: Collection name

        Returns:
            Collection metadata
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    async def list_collections(self) -> list[dict]:
        """List all collections with document counts.

        Returns:
            List of collection metadata dicts
        """
        pass

    @abstractmethod
    async def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ) -> bool:
        """Add documents to a collection.

        Args:
            collection_name: Target collection
            documents: Document texts
            embeddings: Pre-computed embeddings
            metadatas: Document metadata
            ids: Unique document IDs

        Returns:
            True if added successfully
        """
        pass

    @abstractmethod
    async def delete_documents(self, collection_name: str, ids: list[str]) -> bool:
        """Delete documents from a collection.

        Args:
            collection_name: Target collection
            ids: Document IDs to delete

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    async def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Query similar documents.

        Args:
            collection_name: Target collection
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of matching documents with scores
        """
        pass

    @abstractmethod
    async def check_health(self) -> dict:
        """Check vector store health.

        Returns:
            Dict with status and details
        """
        pass
