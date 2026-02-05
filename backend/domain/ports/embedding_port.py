"""Embedding Port - Interface for embedding generation."""

from abc import ABC, abstractmethod


class EmbeddingPort(ABC):
    """Port interface for embedding generation operations."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        pass
