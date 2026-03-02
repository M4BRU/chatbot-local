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


@lru_cache(maxsize=1)
def get_conversation_manager():
    """Singleton ConversationManager — SQLite conversation persistence."""
    from core.conversation_manager import ConversationManager
    return ConversationManager()


@lru_cache(maxsize=1)
def get_catalog_adapter():
    """Singleton CatalogAdapter — SQLite catalog + panier persistence."""
    from backend.adapters.catalog_adapter import CatalogAdapter
    adapter = CatalogAdapter()
    adapter.init_db()
    return adapter


@lru_cache(maxsize=1)
def get_devis_service():
    """Singleton DevisService — tool-calling orchestration for devis mode."""
    from backend.domain.services.devis_service import DevisService
    return DevisService(catalog_adapter=get_catalog_adapter())
