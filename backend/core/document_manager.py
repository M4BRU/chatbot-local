"""
core/document_manager.py — Indexation incrémentale des documents.

Tracking via metadata.json par collection (hash SHA256, date, chunk_ids).
"""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.collection_manager import CollectionManager
from core.parsers import parser_document

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class DocumentManager:
    """Gère l'indexation incrémentale des documents dans les collections."""

    def __init__(self, collection_manager: CollectionManager | None = None):
        self.cm = collection_manager or CollectionManager()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def _metadata_path(self, nom_collection: str) -> Path:
        return self.cm._chemin_collection(nom_collection) / "metadata.json"

    def _charger_metadata(self, nom_collection: str) -> dict:
        chemin = self._metadata_path(nom_collection)
        if chemin.exists():
            return json.loads(chemin.read_text(encoding="utf-8"))
        return {"documents": {}}

    def _sauvegarder_metadata(self, nom_collection: str, metadata: dict) -> None:
        chemin = self._metadata_path(nom_collection)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _calculer_hash(chemin: Path) -> str:
        h = hashlib.sha256()
        with open(chemin, "rb") as f:
            for bloc in iter(lambda: f.read(8192), b""):
                h.update(bloc)
        return h.hexdigest()

    def document_est_indexe(self, nom_collection: str, chemin: Path) -> bool:
        """Vérifie si un document est déjà indexé (même hash SHA256)."""
        chemin = Path(chemin)
        metadata = self._charger_metadata(nom_collection)
        doc_info = metadata["documents"].get(chemin.name)
        if not doc_info:
            return False
        return doc_info["sha256"] == self._calculer_hash(chemin)

    def ajouter_document(self, nom_collection: str, chemin: Path, force: bool = False) -> dict:
        """
        Indexe un document dans une collection.

        Retourne un dict : {"status": "indexed"|"skipped", "chunks": int, "message": str}
        """
        chemin = Path(chemin)

        if not force and self.document_est_indexe(nom_collection, chemin):
            return {
                "status": "skipped",
                "chunks": 0,
                "message": f"{chemin.name} : déjà indexé (hash identique)",
            }

        # Parser le document
        pages = parser_document(chemin)
        if not pages:
            return {
                "status": "skipped",
                "chunks": 0,
                "message": f"{chemin.name} : aucun texte extrait",
            }

        # Découper en chunks
        textes = []
        metadonnees = []
        chunk_ids = []

        for page in pages:
            morceaux = self.splitter.split_text(page.texte)
            for morceau in morceaux:
                cid = str(uuid.uuid4())
                chunk_ids.append(cid)
                textes.append(morceau)
                metadonnees.append({
                    "source": page.source,
                    "page": page.page,
                })

        # Supprimer les anciens chunks de ce document si re-indexation
        metadata = self._charger_metadata(nom_collection)
        doc_info = metadata["documents"].get(chemin.name)
        if doc_info and doc_info.get("chunk_ids"):
            try:
                db = self.cm.get_collection(nom_collection)
                db.delete(ids=doc_info["chunk_ids"])
            except Exception:
                pass

        # Ajouter les nouveaux chunks
        db = self.cm.creer_collection(nom_collection)
        db.add_texts(texts=textes, metadatas=metadonnees, ids=chunk_ids)

        # Mettre à jour le metadata.json
        metadata["documents"][chemin.name] = {
            "sha256": self._calculer_hash(chemin),
            "date": datetime.now().isoformat(),
            "chunk_ids": chunk_ids,
            "nb_chunks": len(chunk_ids),
            "nb_pages": len(pages),
        }
        self._sauvegarder_metadata(nom_collection, metadata)

        return {
            "status": "indexed",
            "chunks": len(chunk_ids),
            "message": f"{chemin.name} : {len(chunk_ids)} chunks indexés ({len(pages)} pages)",
        }

    def supprimer_document(self, nom_collection: str, nom_fichier: str) -> bool:
        """Supprime un document de la collection (chunks + metadata)."""
        metadata = self._charger_metadata(nom_collection)
        doc_info = metadata["documents"].get(nom_fichier)
        if not doc_info:
            return False

        # Supprimer les chunks de ChromaDB
        if doc_info.get("chunk_ids"):
            try:
                db = self.cm.get_collection(nom_collection)
                db.delete(ids=doc_info["chunk_ids"])
            except Exception:
                pass

        # Retirer du metadata
        del metadata["documents"][nom_fichier]
        self._sauvegarder_metadata(nom_collection, metadata)
        return True

    def lister_documents(self, nom_collection: str) -> list[dict]:
        """Liste les documents indexés dans une collection."""
        metadata = self._charger_metadata(nom_collection)
        docs = []
        for nom, info in metadata["documents"].items():
            docs.append({
                "nom": nom,
                "date": info.get("date", ""),
                "nb_chunks": info.get("nb_chunks", 0),
                "nb_pages": info.get("nb_pages", 0),
                "sha256": info.get("sha256", ""),
            })
        return docs
