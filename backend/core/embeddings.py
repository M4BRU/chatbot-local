"""
core/embeddings.py — Configuration Ollama et embeddings.

Source unique de vérité pour le modèle et l'URL du serveur Ollama.

Modèles supportés et leurs préfixes :
  - nomic-embed-text   : doc="search_document: "  query="search_query: "
  - mxbai-embed-large  : doc=""                   query="Represent this sentence for searching relevant passages: "
  - (autres)           : configurable via OLLAMA_EMBED_DOC_PREFIX / OLLAMA_EMBED_QUERY_PREFIX

Mode embeddings :
  - USE_HF_EMBEDDINGS=false (défaut) : embeddings via Ollama (GPU)
  - USE_HF_EMBEDDINGS=true           : embeddings via HuggingFace sentence-transformers (CPU)
    → libère la VRAM pour le LLM, élimine les évictions de modèle (swap de 30s)
    → qualité identique, latence embed ~100-200ms au lieu de 67ms (imperceptible)
"""

import os
import urllib.request
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

# --- Configuration centralisée (variables d'env ou valeurs locales par défaut) ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"

# Mode HuggingFace : embeddings sur CPU, libère la VRAM pour le LLM
USE_HF_EMBEDDINGS = os.environ.get("USE_HF_EMBEDDINGS", "false").lower() == "true"
HF_EMBED_MODEL = os.environ.get("EMBED_HF_MODEL", "mixedbread-ai/mxbai-embed-large-v1")

# Préfixes d'instruction selon le modèle
_DEFAULT_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
}
_defaults = _DEFAULT_PREFIXES.get(EMBEDDING_MODEL, ("", ""))
EMBED_DOC_PREFIX = os.environ.get("OLLAMA_EMBED_DOC_PREFIX", _defaults[0])
EMBED_QUERY_PREFIX = os.environ.get("OLLAMA_EMBED_QUERY_PREFIX", _defaults[1])

_MAX_CHARS = 700  # ~490 tokens max — couvre chunks de 450 tokens même en tableau Docling
                  # (450 tokens × 1.4 char/token = 630 chars, arrondi 700 pour sécurité)
                  # Ancienne valeur 350 tronquait silencieusement ~50% du contenu


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
    Tourne sur GPU via Ollama. Sur petite VRAM (<8 Go), provoque des évictions
    du LLM à chaque appel embed → privilégier HFEmbeddings dans ce cas.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t[:_MAX_CHARS] if len(t) > _MAX_CHARS else t for t in texts]
        if EMBED_DOC_PREFIX:
            texts = [f"{EMBED_DOC_PREFIX}{t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"{EMBED_QUERY_PREFIX}{text}")


class HFEmbeddings(Embeddings):
    """
    Embeddings via HuggingFace sentence-transformers sur CPU.
    Libère intégralement la VRAM pour le LLM — élimine les évictions Ollama.
    Modèle téléchargé depuis HuggingFace Hub au premier démarrage (~700 Mo),
    puis mis en cache dans le volume hf_cache.
    """

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(HF_EMBED_MODEL, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t[:_MAX_CHARS] if len(t) > _MAX_CHARS else t for t in texts]
        if EMBED_DOC_PREFIX:
            texts = [f"{EMBED_DOC_PREFIX}{t}" for t in texts]
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(f"{EMBED_QUERY_PREFIX}{text}", normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Retourne l'instance d'embeddings selon USE_HF_EMBEDDINGS (singleton via lru_cache).
    Sans cache, HuggingFaceEmbeddings rechargerait ~700 Mo depuis le disque à chaque requête.
      - false (défaut) : NomicEmbeddings via Ollama (GPU)
      - true           : HFEmbeddings via sentence-transformers (CPU)
    """
    if USE_HF_EMBEDDINGS:
        return HFEmbeddings()
    return NomicEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
