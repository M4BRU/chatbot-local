"""
core/embeddings.py — Configuration Ollama et embeddings.

Source unique de vérité pour le modèle et l'URL du serveur Ollama.
"""

import os
import urllib.request

from langchain_ollama import OllamaEmbeddings

# --- Configuration centralisée (avec support des variables d'environnement) ---
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


def get_embeddings() -> OllamaEmbeddings:
    """Retourne une instance OllamaEmbeddings configurée avec le modèle dédié."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
