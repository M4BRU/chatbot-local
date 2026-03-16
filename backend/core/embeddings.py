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
  - EMBED_HF_MODEL=BAAI/bge-m3-unsupervised : active bge-m3 (dense 1024-dim + sparse SPLADE)
    → USE_HF_EMBEDDINGS=true requis, EMBED_SPARSE=true pour activer les vecteurs sparse
    → Nécessite HF_HUB_OFFLINE=0 au premier démarrage pour télécharger le modèle (~1.1GB)
"""

import logging
import os
import urllib.request
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

# --- Configuration centralisée (variables d'env ou valeurs locales par défaut) ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"

# Mode HuggingFace : embeddings sur CPU, libère la VRAM pour le LLM
USE_HF_EMBEDDINGS = os.environ.get("USE_HF_EMBEDDINGS", "false").lower() == "true"
HF_EMBED_MODEL = os.environ.get("EMBED_HF_MODEL", "mixedbread-ai/mxbai-embed-large-v1")

# bge-m3 : vecteurs sparse SPLADE en plus du dense (pour Qdrant hybrid search)
EMBED_SPARSE = os.environ.get("EMBED_SPARSE", "false").lower() == "true"
_IS_BGE_M3 = "bge-m3" in HF_EMBED_MODEL.lower()

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


# ── bge-m3 : dense + sparse (Group B) ─────────────────────────────────────────
# Singleton partagé entre BGEM3Embeddings et BGEM3SparseEmbeddings.
# Le modèle (~1.1GB) est lourd — on le charge une seule fois en mémoire.
_bge_m3_model = None


def _get_bge_m3_model():
    """Charge BGEM3FlagModel (singleton). Échec si FlagEmbedding absent ou modèle non téléchargé."""
    global _bge_m3_model
    if _bge_m3_model is not None:
        return _bge_m3_model
    from FlagEmbedding import BGEM3FlagModel
    logger.info(f"bge-m3 : chargement du modèle {HF_EMBED_MODEL} (CPU)…")
    _bge_m3_model = BGEM3FlagModel(HF_EMBED_MODEL, use_fp16=False, device="cpu")
    logger.info(f"bge-m3 initialisé : {HF_EMBED_MODEL} (CPU, dense+sparse disponibles)")
    return _bge_m3_model


class BGEM3Embeddings(Embeddings):
    """
    Embeddings denses bge-m3 via FlagEmbedding (CPU, 1024-dim).
    Compatible avec ChromaDB et Qdrant dense index.
    bge-m3-unsupervised : identique à bge-m3 mais sans supervision de tâche spécifique.
    Dimension : 1024 — identique à mxbai-embed-large-v1 (même index Chroma utilisable).
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t[:_MAX_CHARS] if len(t) > _MAX_CHARS else t for t in texts]
        model = _get_bge_m3_model()
        result = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return result["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list[float]:
        model = _get_bge_m3_model()
        result = model.encode([text[:_MAX_CHARS]], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return result["dense_vecs"][0].tolist()


class BGEM3SparseEmbeddings:
    """
    Embeddings sparse bge-m3 (SPLADE lexical weights) via FlagEmbedding.
    Compatible avec Qdrant sparse index natif.
    Retourne des SparseVector(indices, values) — format attendu par langchain-qdrant.
    """

    def _to_sparse_vector(self, weights: dict):
        try:
            from langchain_qdrant.sparse_embeddings import SparseVector
        except ImportError:
            from dataclasses import make_dataclass
            SparseVector = make_dataclass("SparseVector", [("indices", list), ("values", list)])
        indices = [int(k) for k in weights.keys()]
        values = [float(v) for v in weights.values()]
        return SparseVector(indices=indices, values=values)

    def embed_documents(self, texts: list[str]) -> list:
        texts = [t[:_MAX_CHARS] if len(t) > _MAX_CHARS else t for t in texts]
        model = _get_bge_m3_model()
        result = model.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        return [self._to_sparse_vector(w) for w in result["lexical_weights"]]

    def embed_query(self, text: str):
        model = _get_bge_m3_model()
        result = model.encode([text[:_MAX_CHARS]], return_dense=False, return_sparse=True, return_colbert_vecs=False)
        return self._to_sparse_vector(result["lexical_weights"][0])


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Retourne l'instance d'embeddings selon USE_HF_EMBEDDINGS (singleton via lru_cache).
    Sans cache, HuggingFaceEmbeddings rechargerait ~700 Mo depuis le disque à chaque requête.
      - false (défaut) : NomicEmbeddings via Ollama (GPU)
      - true + bge-m3  : BGEM3Embeddings via FlagEmbedding (CPU, 1024-dim dense)
      - true           : HFEmbeddings via sentence-transformers (CPU)
    """
    if USE_HF_EMBEDDINGS and _IS_BGE_M3:
        return BGEM3Embeddings()
    if USE_HF_EMBEDDINGS:
        return HFEmbeddings()
    return NomicEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_sparse_embeddings():
    """
    Retourne BGEM3SparseEmbeddings si EMBED_SPARSE=true et bge-m3 actif, None sinon.
    Utilisé par QdrantCollectionManager pour le hybrid search natif.
    """
    if EMBED_SPARSE and _IS_BGE_M3 and USE_HF_EMBEDDINGS:
        logger.info("bge-m3 sparse embeddings activés (SPLADE)")
        return BGEM3SparseEmbeddings()
    return None
