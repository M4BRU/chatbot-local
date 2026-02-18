"""
core/embeddings.py — Configuration Ollama et embeddings.

Source unique de vérité pour le modèle et l'URL du serveur Ollama.

Modèles supportés et leurs préfixes :
  - nomic-embed-text   : doc="search_document: "  query="search_query: "
  - mxbai-embed-large  : doc=""                   query="Represent this sentence for searching relevant passages: "
  - (autres)           : configurable via OLLAMA_EMBED_DOC_PREFIX / OLLAMA_EMBED_QUERY_PREFIX
"""

import os
import urllib.request

from langchain_ollama import OllamaEmbeddings

# --- Configuration centralisée (variables d'env ou valeurs locales par défaut) ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"

# Préfixes d'instruction selon le modèle
_DEFAULT_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
}
_defaults = _DEFAULT_PREFIXES.get(EMBEDDING_MODEL, ("", ""))
EMBED_DOC_PREFIX = os.environ.get("OLLAMA_EMBED_DOC_PREFIX", _defaults[0])
EMBED_QUERY_PREFIX = os.environ.get("OLLAMA_EMBED_QUERY_PREFIX", _defaults[1])


def verifier_ollama() -> bool:
    """Vérifie que le serveur Ollama est accessible."""
    try:
        urllib.request.urlopen(OLLAMA_BASE_URL, timeout=5)
        return True
    except Exception:
        return False


class NomicEmbeddings(OllamaEmbeddings):
    """
    OllamaEmbeddings avec préfixes d'instruction configurables.

    Les modèles instruction-tuned nécessitent des préfixes différents selon
    leur entraînement. Les préfixes sont lus depuis les variables d'env
    OLLAMA_EMBED_DOC_PREFIX et OLLAMA_EMBED_QUERY_PREFIX (ou auto-détectés
    selon OLLAMA_EMBED_MODEL).
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Tronque les textes trop longs avant d'ajouter le préfixe.
        # mxbai-embed-large : 512 tokens max.
        # Tableaux markdown (|, chiffres) : ratio ~2 chars/token → 1200 chars = 600 tokens, trop.
        # 500 chars garantit <250 tokens même pour le pire cas tabulaire.
        MAX_CHARS = 500
        texts = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]
        if EMBED_DOC_PREFIX:
            texts = [f"{EMBED_DOC_PREFIX}{t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"{EMBED_QUERY_PREFIX}{text}")


def get_embeddings() -> NomicEmbeddings:
    """Retourne une instance NomicEmbeddings configurée avec le modèle dédié."""
    return NomicEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
