"""
core/embeddings.py — Configuration Ollama et embeddings.

Source unique de vérité pour le modèle et l'URL du serveur Ollama.
"""

import os
import urllib.request

from langchain_ollama import OllamaEmbeddings

# --- Configuration centralisée (variables d'env ou valeurs locales par défaut) ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"


def verifier_ollama() -> bool:
    """Vérifie que le serveur Ollama est accessible."""
    try:
        urllib.request.urlopen(OLLAMA_BASE_URL, timeout=5)
        return True
    except Exception:
        return False


class NomicEmbeddings(OllamaEmbeddings):
    """
    OllamaEmbeddings avec préfixes d'instruction pour nomic-embed-text.

    nomic-embed-text est un modèle instruction-tuned : sans préfixes, les
    vecteurs de requête et de document sont mal alignés (scores > 0.7 même
    pour des correspondances exactes).

    Préfixes officiels :
        - documents indexés  → "search_document: <texte>"
        - requêtes           → "search_query: <texte>"
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")


def get_embeddings() -> NomicEmbeddings:
    """Retourne une instance NomicEmbeddings configurée avec le modèle dédié."""
    return NomicEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
