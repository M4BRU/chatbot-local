"""Dependency injection wiring for FastAPI."""

from functools import lru_cache

from backend.config.settings import Settings


@lru_cache
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()


@lru_cache(maxsize=1)
def get_collection_manager():
    """Singleton CollectionManager — une seule connexion HTTP vers ChromaDB."""
    from core.collection_manager import CollectionManager
    return CollectionManager()
