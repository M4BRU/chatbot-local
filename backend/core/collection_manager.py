"""
core/collection_manager.py — Gestion multi-collections ChromaDB via HTTP.

Se connecte au service ChromaDB via HTTP (mode microservices).
Configuration par variables d'environnement :
    CHROMA_HOST  : hôte ChromaDB (défaut: localhost)
    CHROMA_PORT  : port ChromaDB (défaut: 8100 en local, 8000 dans Docker)
"""

import os

import chromadb
from langchain_chroma import Chroma

from core.embeddings import get_embeddings

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8100"))


class CollectionManager:
    """Gère les collections ChromaDB via le client HTTP."""

    def __init__(self, host: str | None = None, port: int | None = None):
        self.host = host or CHROMA_HOST
        self.port = port or CHROMA_PORT
        self._client = chromadb.HttpClient(host=self.host, port=self.port)

    def collection_existe(self, nom: str) -> bool:
        """Vérifie si une collection existe."""
        try:
            self._client.get_collection(nom)
            return True
        except Exception:
            return False

    def creer_collection(self, nom: str) -> Chroma:
        """Crée (ou ouvre) une collection ChromaDB."""
        return Chroma(
            client=self._client,
            collection_name=nom,
            embedding_function=get_embeddings(),
        )

    def get_collection(self, nom: str) -> Chroma:
        """Retourne une collection existante."""
        if not self.collection_existe(nom):
            raise ValueError(f"Collection '{nom}' introuvable.")
        return Chroma(
            client=self._client,
            collection_name=nom,
            embedding_function=get_embeddings(),
        )

    def lister_collections(self) -> list[str]:
        """Liste toutes les collections disponibles."""
        return sorted(c.name for c in self._client.list_collections())

    def supprimer_collection(self, nom: str) -> None:
        """Supprime une collection ChromaDB."""
        self._client.delete_collection(nom)
