"""
core — Modules backend du chatbot RAG multi-collections.
"""

from core.embeddings import get_embeddings, verifier_ollama, OLLAMA_MODEL, OLLAMA_BASE_URL
from core.parsers import parser_document, extensions_supportees
from core.collection_manager import CollectionManager
from core.document_manager import DocumentManager
from core.search import RAGEngine

__all__ = [
    "get_embeddings",
    "verifier_ollama",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "parser_document",
    "extensions_supportees",
    "CollectionManager",
    "DocumentManager",
    "RAGEngine",
]
