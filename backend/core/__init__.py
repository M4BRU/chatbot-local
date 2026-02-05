"""
core — Modules backend du chatbot RAG multi-collections.
"""

from core.collection_manager import CollectionManager
from core.document_manager import DocumentManager
from core.embeddings import OLLAMA_BASE_URL, OLLAMA_MODEL, get_embeddings, verifier_ollama
from core.parsers import extensions_supportees, parser_document
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
