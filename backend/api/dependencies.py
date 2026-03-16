"""Dependency injection wiring for FastAPI."""

import os
from functools import lru_cache

from backend.config.settings import Settings

# VECTOR_DB=chroma (défaut) ou VECTOR_DB=qdrant (Group B)
_VECTOR_DB = os.environ.get("VECTOR_DB", "chroma").lower()


@lru_cache
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()


@lru_cache(maxsize=1)
def get_collection_manager():
    """
    Singleton CollectionManager — factory selon VECTOR_DB.
      VECTOR_DB=chroma (défaut) : CollectionManager (ChromaDB HTTP)
      VECTOR_DB=qdrant          : QdrantCollectionManager (Qdrant HTTP)
    """
    if _VECTOR_DB == "qdrant":
        from core.qdrant_collection_manager import QdrantCollectionManager
        return QdrantCollectionManager()
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
def get_document_manager():
    """Singleton DocumentManager — indexation + pipeline version check."""
    from core.document_manager import DocumentManager
    return DocumentManager(collection_manager=get_collection_manager())


@lru_cache(maxsize=1)
def get_excel_collection_adapter():
    """Singleton ExcelCollectionAdapter — pipeline SQL déterministe pour Excel RAG."""
    from backend.adapters.excel_collection_adapter import ExcelCollectionAdapter
    adapter = ExcelCollectionAdapter()
    adapter.init_db()
    return adapter


@lru_cache(maxsize=1)
def get_devis_service():
    """Singleton DevisService — tool-calling orchestration for devis mode."""
    from backend.domain.services.devis_service import DevisService
    return DevisService(
        catalog_adapter=get_catalog_adapter(),
        excel_adapter=get_excel_collection_adapter(),
    )
