"""
core/collection_manager.py — Gestion multi-collections ChromaDB.

Chaque collection est stockée dans un sous-dossier distinct :
    ./chroma_db/{nom_collection}/
"""

import shutil
from pathlib import Path

from langchain_chroma import Chroma

from core.embeddings import get_embeddings

CHROMA_BASE_DIR = Path("./chroma_db")


class CollectionManager:
    """Gère les collections ChromaDB (CRUD)."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else CHROMA_BASE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _chemin_collection(self, nom: str) -> Path:
        return self.base_dir / nom

    def collection_existe(self, nom: str) -> bool:
        """Vérifie si une collection existe."""
        chemin = self._chemin_collection(nom)
        return chemin.exists() and any(chemin.iterdir())

    def creer_collection(self, nom: str) -> Chroma:
        """Crée (ou ouvre) une collection ChromaDB."""
        chemin = self._chemin_collection(nom)
        chemin.mkdir(parents=True, exist_ok=True)
        return Chroma(
            persist_directory=str(chemin),
            embedding_function=get_embeddings(),
        )

    def get_collection(self, nom: str) -> Chroma:
        """Retourne une collection existante."""
        if not self.collection_existe(nom):
            raise ValueError(f"Collection '{nom}' introuvable.")
        return Chroma(
            persist_directory=str(self._chemin_collection(nom)),
            embedding_function=get_embeddings(),
        )

    def lister_collections(self) -> list[str]:
        """Liste toutes les collections disponibles."""
        if not self.base_dir.exists():
            return []
        collections = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir() and any(d.iterdir()):
                collections.append(d.name)
        return collections

    def supprimer_collection(self, nom: str) -> None:
        """Supprime une collection et tous ses fichiers."""
        chemin = self._chemin_collection(nom)
        if chemin.exists():
            shutil.rmtree(chemin)
