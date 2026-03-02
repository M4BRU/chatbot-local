# Audit de code — chatbot-local RAG backend

Tu es Claude Opus, tu effectues un audit de code approfondi et rigoureux.
Ce projet est un chatbot RAG local (Retrieval-Augmented Generation) avec architecture hexagonale partielle, FastAPI, ChromaDB, Ollama, et un frontend Next.js.

---

## CONTEXTE DU PROJET

**Stack technique :**
- Backend : FastAPI + Python 3.12, architecture hexagonale (partielle)
- LLM : Ollama (llama3.1:8b) en local via GPU NVIDIA
- Embeddings : HuggingFace sentence-transformers (mxbai-embed-large-v1) sur CPU OU Ollama
- Vector store : ChromaDB (mode HTTP)
- Reranker : BAAI/bge-reranker-v2-m3 (cross-encoder) ou ColBERT (optionnel)
- Hybrid search : BM25 (rank_bm25) + vector search, fusion RRF
- Parsers : Docling (PDF/DOCX/XLSX) + fallbacks PyMuPDF4LLM / python-docx / pandas
- Frontend : Next.js 14, SSE streaming, React
- Infrastructure : Docker Compose, GPU passthrough Ollama

**Architecture :**
```
backend/
  api/
    routes/         # FastAPI route handlers
    dependencies.py # DI wiring (lru_cache settings)
  config/settings.py
  domain/
    models/         # Pydantic domain models
    ports/          # Interfaces (LLMPort, EmbeddingPort, VectorStorePort) — NON IMPLÉMENTÉES
  core/             # Logique métier principale (legacy-wrap)
    collection_manager.py
    document_manager.py
    embeddings.py
    parsers.py
    search.py
  main.py
frontend/
  app/
    components/Chat.tsx
    admin/page.tsx
    lib/api.ts
```

**PYTHONPATH Docker :** `PYTHONPATH=/app/backend:/app`
Les routes importent `from core.xxx` (pas `from backend.core.xxx`), résolu par le PYTHONPATH.

---

## CONTENU INTÉGRAL DES FICHIERS

---

### `/home/lucmc94/chatbot-local/backend/core/embeddings.py`

```python
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

_MAX_CHARS = 350  # ~250 tokens max, couvre tableaux Markdown Docling (~1.4 char/token)


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
        from langchain_huggingface import HuggingFaceEmbeddings
        self._model = HuggingFaceEmbeddings(
            model_name=HF_EMBED_MODEL,
            model_kwargs={"device": "cpu", "low_cpu_mem_usage": False},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t[:_MAX_CHARS] if len(t) > _MAX_CHARS else t for t in texts]
        if EMBED_DOC_PREFIX:
            texts = [f"{EMBED_DOC_PREFIX}{t}" for t in texts]
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(f"{EMBED_QUERY_PREFIX}{text}")


def get_embeddings() -> Embeddings:
    """
    Retourne l'instance d'embeddings selon USE_HF_EMBEDDINGS :
      - false (défaut) : NomicEmbeddings via Ollama (GPU)
      - true           : HFEmbeddings via sentence-transformers (CPU)
    """
    if USE_HF_EMBEDDINGS:
        return HFEmbeddings()
    return NomicEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
```

---

### `/home/lucmc94/chatbot-local/backend/core/collection_manager.py`

```python
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
```

---

### `/home/lucmc94/chatbot-local/backend/core/search.py`

```python
"""
core/search.py — RAGEngine : recherche similarité + génération Ollama streaming.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from core.collection_manager import CollectionManager
from core.embeddings import OLLAMA_API_GENERATE, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# Prompt par défaut générique
PROMPT_DEFAUT = """Tu es un assistant intelligent. Utilise le contexte ci-dessous pour répondre à la question.

Contexte :
{context}

Question : {question}

Réponds de manière précise et concise. Si l'information n'est pas dans le contexte, dis-le clairement."""

# Prompt spécialisé VLM Robotics
PROMPT_VLM_ROBOTICS = """Tu es un assistant commercial expert pour VLM Robotics, constructeur de machines-outils robotisées pour la fabrication hybride XXL (Machine Tool Builder for XXL Hybrid Manufacturing).

Ton expertise couvre :
- La gamme complète : COMPAQT (XL, entrée de gamme), SOLO (XXL mono-robot), GEMINI (XXL bi-robot, la plus avancée), HYMANCO (unité mobile containerisée)
- Les multiples technologies intégrées : WAAM (arc électrique CMT Fronius), fabrication additive hybride laser (poudre et fil), Cold Spray, FSW, usinage, CND, scan/métrologie, collage, polymère FDM
- L'expertise VLM : Direct Control, commande numérique CNC Siemens, logiciel NX (CAO, CAM, jumeau numérique), continuité numérique, Industry 4.0
- Le positionnement hybride : les machines combinent plusieurs procédés sur une même plateforme (ex : fabrication additive + usinage + contrôle)
- Les secteurs : ASD, Ferroviaire, Naval, Énergie, MRO, Fonderie, Outillage, Formation, Recherche, Offshore

Contexte disponible :
{context}

Question client : {question}

Consignes de réponse :
- Réponds en français, de manière professionnelle et structurée.
- Cite toujours la source (nom de la machine, référence brochure, numéro de page).
- Si le contexte permet de recommander une machine spécifique, explique pourquoi elle convient au besoin.
- Si l'information n'est pas dans le contexte fourni, dis-le clairement : « Je n'ai pas trouvé cette information dans la documentation disponible. »
- Ne jamais inventer de spécifications techniques."""

# Prompts nommés disponibles
PROMPTS = {
    "defaut": PROMPT_DEFAUT,
    "vlm_robotics": PROMPT_VLM_ROBOTICS,
    "vlm": PROMPT_VLM_ROBOTICS,  # alias : collection "vlm" → prompt VLM Robotics
}

NB_CHUNKS_RECHERCHE = 6   # fallback si collection vide
CHUNK_SIZE_APPROX = 1000  # doit correspondre à document_manager.CHUNK_SIZE
NUM_CTX_MIN = 4096
NUM_CTX_MAX = 32768

# ── Metadata filtering ────────────────────────────────────────────────────────
_MACHINES_CONNUES = ["GEMINI", "SOLO", "COMPAQT", "HYMANCO"]

# ── Reranker config ───────────────────────────────────────────────────────────
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_CANDIDATS_MULT = 3   # récupère k×3 candidats avant reranking
RERANKER_CANDIDATS_MAX = 25   # plafond pour éviter un contexte trop large
_reranker_instance = None

# ── ColBERT config (RAGatouille) ──────────────────────────────────────────────
USE_COLBERT = os.environ.get("USE_COLBERT", "false").lower() == "true"
COLBERT_MODEL = os.environ.get("COLBERT_MODEL", "colbert-ir/colbertv2.0")
_colbert_instance = None

# ── Session scoping ───────────────────────────────────────────────────────────
SCOPING_BOOST = 2.0   # Facteur multiplicatif appliqué aux sources identifiées

_MOTS_EXCLUS_SCOPING = {
    "GEMINI", "SOLO", "COMPAQT", "HYMANCO",
    "PLAN", "OFFRE", "POUR", "DANS", "AVEC", "NOUS",
    "VOUS", "SONT", "SERA", "LEUR", "CETTE", "AUSSI",
    "LISTE", "GENIE", "CIVIL", "LASER", "TECHNIQUE",
    "USER", "ASSISTANT", "FROM", "WITH", "THAT",
    "THIS", "WHAT", "ABOUT", "CONTEXT",
}


def _extraire_identifiants_session(history: str) -> set[str]:
    if not history:
        return set()
    tokens = re.findall(r'\b[A-Z]{4,}\b', history)
    return {t for t in tokens if t not in _MOTS_EXCLUS_SCOPING}


def _booster_sources_session(resultats: list, identifiants: set[str]) -> list:
    if not identifiants:
        return resultats

    boosted = []
    for doc, score in resultats:
        source = doc.metadata.get("source", "").upper()
        if any(ident in source for ident in identifiants):
            logger.info(
                f"Session scoping : boost ×{SCOPING_BOOST} "
                f"sur '{doc.metadata.get('source')}' (identifiants: {identifiants})"
            )
            boosted.append((doc, score * SCOPING_BOOST))
        else:
            boosted.append((doc, score))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


def _deduplicater_pdf_docx(resultats: list) -> list:
    seen: dict[tuple, tuple[float, int]] = {}

    for i, (doc, score) in enumerate(resultats):
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "?")
        base = re.sub(r'\.(pdf|docx|doc|txt)$', '', source, flags=re.IGNORECASE)
        key = (base, page)

        if key not in seen or score > seen[key][0]:
            seen[key] = (score, i)

    kept_indices = {idx for _, idx in seen.values()}
    deduped = [r for i, r in enumerate(resultats) if i in kept_indices]

    nb_removed = len(resultats) - len(deduped)
    if nb_removed:
        logger.info(f"Déduplication PDF/DOCX : {nb_removed} doublon(s) supprimé(s)")

    return deduped


# ── Hybrid search (BM25 + vector) ────────────────────────────────────────────
USE_HYBRID_SEARCH = os.environ.get("USE_HYBRID_SEARCH", "true").lower() == "true"
RRF_K = 60
_bm25_cache: dict[str, tuple[int, object]] = {}  # {collection: (nb_chunks, index)}

_stemmer = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem.snowball import FrenchStemmer
            _stemmer = FrenchStemmer()
        except Exception as e:
            logger.warning(f"Stemmer NLTK indisponible ({e}) — stemming désactivé")
    return _stemmer


def _stemmer_tokens(tokens: list[str]) -> list[str]:
    stemmer = _get_stemmer()
    if stemmer is None:
        return tokens
    return [stemmer.stem(t) for t in tokens]


class _BM25CollectionIndex:
    """Index BM25 sur les chunks d'une collection ChromaDB."""

    def __init__(self, docs: list, texts: list):
        from rank_bm25 import BM25Okapi
        self._docs = docs
        tokenized = [_stemmer_tokens(re.findall(r'\w+', t.lower())) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int, filtre: dict | None = None) -> list:
        tokens = _stemmer_tokens(re.findall(r'\w+', query.lower()))
        raw_scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)
        machine_filtre = None
        if filtre:
            machine_filtre = filtre.get("machine", {}).get("$eq")
        results = []
        for i in ranked:
            doc = self._docs[i]
            if machine_filtre and doc.metadata.get("machine") != machine_filtre:
                continue
            results.append((doc, float(raw_scores[i])))
            if len(results) >= k:
                break
        return results


def _get_or_build_bm25(db, collection_name: str) -> "_BM25CollectionIndex | None":
    if not USE_HYBRID_SEARCH:
        return None
    try:
        nb_chunks = db._collection.count()
        cached = _bm25_cache.get(collection_name)
        if cached and cached[0] == nb_chunks:
            return cached[1]
        logger.info(f"BM25 : construction de l'index '{collection_name}' ({nb_chunks} chunks)…")
        result = db._collection.get(include=["documents", "metadatas"])
        texts = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not texts:
            return None
        from langchain_core.documents import Document
        docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas)]
        index = _BM25CollectionIndex(docs, texts)
        _bm25_cache[collection_name] = (nb_chunks, index)
        logger.info(f"BM25 : index prêt ({len(texts)} chunks)")
        return index
    except Exception as e:
        logger.warning(f"BM25 index indisponible ({e}) — hybrid search désactivé")
        return None


def _rrf_fusion(vector_results: list, bm25_results: list, k_final: int) -> list:
    def uid(doc) -> str:
        return doc.page_content[:150]  # fingerprint unique par chunk

    scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}

    for rank, (doc, _) in enumerate(vector_results):
        u = uid(doc)
        scores[u] = scores.get(u, 0.0) + 1.0 / (RRF_K + rank + 1)
        doc_map[u] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        u = uid(doc)
        scores[u] = scores.get(u, 0.0) + 1.0 / (RRF_K + rank + 1)
        doc_map[u] = doc

    sorted_uids = sorted(scores, key=lambda u: scores[u], reverse=True)
    return [(doc_map[u], scores[u]) for u in sorted_uids[:k_final]]


# ── Keyword fallback (where_document) ─────────────────────────────────────────
KEYWORD_FALLBACK_THRESHOLD = float(os.environ.get("KEYWORD_FALLBACK_THRESHOLD", "0.1"))

_STOPWORDS_FALLBACK = {
    "les", "des", "pour", "dans", "avec", "sur", "par", "une", "qui", "que",
    "est", "son", "ses", "moi", "lui", "leur", "tout", "cette", "aussi",
    "donne", "mois", "references", "documents", "trouve", "trouver",
    "parle", "concernant", "cela", "ceci", "avoir", "etre", "faire",
    "peux", "mots", "toute", "base", "donnees", "infos", "informations",
    "passages", "concernant",
    "the", "and", "for", "with", "that", "this", "from", "about",
}


def _keyword_fallback_search(db, question: str, k: int) -> list:
    from langchain_core.documents import Document

    tokens = re.findall(r'\w+', question.lower())
    mots_cles = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS_FALLBACK]
    if not mots_cles:
        return []

    resultats = []
    vus: set[str] = set()

    for mot in mots_cles[:3]:
        variantes_morpho = {mot}
        if mot.endswith('s') and len(mot) > 3:
            variantes_morpho.add(mot[:-1])
        elif mot.endswith('aux') and len(mot) > 4:
            variantes_morpho.add(mot[:-3] + 'al')
        else:
            variantes_morpho.add(mot + 's')
        variantes = {v for base in variantes_morpho for v in (base, base.upper(), base.capitalize())}
        for variant in variantes:
            try:
                result = db._collection.get(
                    where_document={"$contains": variant},
                    include=["documents", "metadatas"],
                )
                docs = result.get("documents") or []
                metas = result.get("metadatas") or []
                for text, meta in zip(docs, metas):
                    uid = text[:120]
                    if uid not in vus:
                        vus.add(uid)
                        resultats.append(
                            (Document(page_content=text, metadata=meta or {}), 0.5)
                        )
                        if len(resultats) >= k:
                            break
            except Exception as e:
                logger.debug(f"where_document('{variant}') échoué : {e}")
            if len(resultats) >= k:
                break
        if len(resultats) >= k:
            break

    return resultats


def _extraire_filtre_question(question: str) -> dict | None:
    q_upper = question.upper()
    machines_trouvees = [m for m in _MACHINES_CONNUES if m in q_upper]
    if len(machines_trouvees) == 1:
        return {"machine": {"$eq": machines_trouvees[0]}}
    return None


def _get_reranker():
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance
    if not USE_RERANKER:
        return None
    try:
        import torch
        from FlagEmbedding import FlagReranker
        use_fp16 = torch.cuda.is_available()
        _reranker_instance = FlagReranker(RERANKER_MODEL, use_fp16=use_fp16)
        logger.info(f"Reranker initialisé : {RERANKER_MODEL} (fp16={use_fp16})")
        return _reranker_instance
    except Exception as e:
        logger.warning(f"Reranker indisponible ({e}) — désactivé")
        return None


def _get_colbert():
    global _colbert_instance
    if _colbert_instance is not None:
        return _colbert_instance
    if not USE_COLBERT:
        return None
    try:
        from ragatouille import RAGPretrainedModel
        _colbert_instance = RAGPretrainedModel.from_pretrained(COLBERT_MODEL)
        logger.info(f"ColBERT initialisé : {COLBERT_MODEL}")
        return _colbert_instance
    except Exception as e:
        logger.warning(f"ColBERT indisponible ({e}) — désactivé")
        return None


def _appliquer_colbert(colbert, question: str, resultats: list, top_k: int) -> list:
    if not resultats:
        return resultats
    try:
        docs_text = [doc.page_content for doc, _ in resultats]
        ranked = colbert.rerank(query=question, documents=docs_text, k=min(top_k, len(docs_text)))
        content_to_original = {doc.page_content: (doc, score) for doc, score in resultats}
        reranked = []
        for item in ranked:
            original = content_to_original.get(item["content"])
            if original:
                reranked.append((original[0], float(item["score"])))
        return reranked
    except Exception as e:
        logger.warning(f"ColBERT reranking échoué ({e}) — résultats originaux conservés")
        return resultats[:top_k]


def _appliquer_reranker(reranker, question: str, resultats: list, top_k: int) -> list:
    try:
        pairs = [[question, doc.page_content] for doc, _ in resultats]
        scores = reranker.compute_score(pairs, normalize=True, max_length=512)
        indexes_tries = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = [(resultats[i][0], float(scores[i])) for i in indexes_tries[:top_k]]
        return top
    except Exception as e:
        logger.warning(f"Reranker échoué ({e}) — résultats originaux conservés")
        return resultats[:top_k]


def _reformuler_question(question: str, history: str) -> str:
    if not history:
        return question

    prompt = (
        "Tu es un moteur de réécriture de requête pour un système RAG.\n"
        "À partir de l'historique de conversation et de la question actuelle, "
        "génère une requête de recherche autonome, courte et riche en mots-clés.\n"
        "La requête doit permettre de retrouver les bons documents même sans contexte.\n"
        "Réponds UNIQUEMENT avec la requête réécrite, sans explication ni ponctuation finale.\n\n"
        f"Historique :\n{history}\n\n"
        f"Question actuelle : {question}\n\n"
        "Requête de recherche :"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 2048},
    }

    try:
        resp = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=15)
        resp.raise_for_status()
        rewritten = resp.json().get("response", "").strip()
        if rewritten:
            logger.info(f"Query rewriting : '{question}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewriting échoué ({e}) — question originale conservée")

    return question


def _charger_prompts_json() -> dict:
    """Charge les prompts supplémentaires depuis prompts.json s'il existe."""
    chemin = Path("prompts.json")
    if chemin.exists():
        try:
            return json.loads(chemin.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_prompt(nom: str) -> str:
    """Retourne un prompt par son nom (built-in ou depuis prompts.json)."""
    if nom in PROMPTS:
        return PROMPTS[nom]
    customs = _charger_prompts_json()
    if nom in customs:
        return customs[nom]
    return PROMPTS["defaut"]


class RAGEngine:
    """Moteur RAG : recherche de similarité + génération Ollama."""

    def __init__(self, nom_collection: str, prompt_name: str = "defaut",
                 collection_manager: CollectionManager | None = None):
        self.cm = collection_manager or CollectionManager()
        self.nom_collection = nom_collection
        self.prompt_template = get_prompt(prompt_name)
        self.db = self.cm.get_collection(nom_collection)

    def _adapter_parametres(self) -> tuple[int, int]:
        try:
            nb_chunks = self.db._collection.count()
        except Exception:
            nb_chunks = 0

        if nb_chunks == 0:
            return NB_CHUNKS_RECHERCHE, NUM_CTX_MIN

        k = max(6, min(nb_chunks // 8, 20))
        tokens_par_chunk = CHUNK_SIZE_APPROX // 4   # ~250 tokens
        overhead = 1200
        num_ctx = k * tokens_par_chunk + overhead
        num_ctx = max(NUM_CTX_MIN, min(num_ctx, NUM_CTX_MAX))
        return k, num_ctx

    def rechercher(self, question: str, k: int = NB_CHUNKS_RECHERCHE,
                   history: str = "") -> tuple[str, list[dict]]:
        # 1. Pré-filtrage metadata
        filtre = _extraire_filtre_question(question)

        # 2. Nombre de candidats
        reranker = _get_reranker()
        k_candidats = min(k * RERANKER_CANDIDATS_MULT, RERANKER_CANDIDATS_MAX) if reranker else k

        # 3. Recherche vectorielle
        try:
            resultats = self.db.similarity_search_with_score(question, k=k_candidats, filter=filtre)
            if not resultats and filtre:
                resultats = self.db.similarity_search_with_score(question, k=k_candidats)
        except Exception as e:
            logger.warning(f"Recherche avec filtre échouée ({e}) — retry sans filtre")
            resultats = self.db.similarity_search_with_score(question, k=k_candidats)

        # 4. Hybrid BM25 — fusion RRF
        bm25_index = _get_or_build_bm25(self.db, self.nom_collection)
        if bm25_index:
            bm25_results = bm25_index.search(question, k=k_candidats, filtre=filtre)
            resultats = _rrf_fusion(resultats, bm25_results, k_final=k_candidats)

        # 5. Reranking
        colbert = _get_colbert()
        if colbert and len(resultats) > k:
            resultats = _appliquer_colbert(colbert, question, resultats, top_k=k)
        elif reranker and len(resultats) > k:
            resultats = _appliquer_reranker(reranker, question, resultats, top_k=k)

        # 5b. Keyword fallback
        score_best = resultats[0][1] if resultats else 0.0
        if score_best < KEYWORD_FALLBACK_THRESHOLD:
            fallback = _keyword_fallback_search(self.db, question, k=k)
            if fallback:
                resultats = fallback + [r for r in resultats if r not in fallback]

        # 6. Seuil reranker relatif
        if reranker and resultats:
            score_max_r = resultats[0][1]
            seuil_relatif = score_max_r * 0.1
            resultats = [(doc, s) for doc, s in resultats if s >= seuil_relatif]

        # 6b. Seuil absolu
        resultats = [(doc, s) for doc, s in resultats if s >= 0.01]

        # 7. Déduplication PDF/DOCX
        resultats = _deduplicater_pdf_docx(resultats)

        # 8. Formater les résultats
        contexte_parts = []
        sources = []
        sources_vues = set()

        for doc, score in resultats:
            contexte_parts.append(doc.page_content)
            cle_source = f"{doc.metadata.get('source', 'Inconnu')} - p.{doc.metadata.get('page', '?')}"
            if cle_source not in sources_vues:
                sources_vues.add(cle_source)
                sources.append({
                    "fichier": doc.metadata.get("source", "Inconnu"),
                    "page": doc.metadata.get("page", "?"),
                    "score": round(score, 3),
                })

        contexte = "\n\n---\n\n".join(contexte_parts)
        return contexte, sources

    def generer_avec_sources(self, question: str, stream: bool = True, history: str = "") -> dict:
        k, num_ctx = self._adapter_parametres()
        query_recherche = _reformuler_question(question, history)
        contexte, sources = self.rechercher(query_recherche, k=k, history=history)

        if history:
            contexte = f"Historique de conversation:\n{history}\n\n---\n\n{contexte}"

        prompt = self.prompt_template.format(context=contexte, question=question)
        reponse = self._appeler_ollama(prompt, stream=stream, num_ctx=num_ctx)
        return {"reponse": reponse, "sources": sources}

    @staticmethod
    def _appeler_ollama(prompt: str, stream: bool = True, num_ctx: int = NUM_CTX_MIN):
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.3,
                "num_ctx": num_ctx,
            },
        }

        try:
            reponse = requests.post(
                OLLAMA_API_GENERATE,
                json=payload,
                stream=stream,
                timeout=300,
            )
            reponse.raise_for_status()
        except requests.ConnectionError:
            msg = "Impossible de contacter Ollama. Vérifiez qu'il est lancé avec `ollama serve`."
            if stream:
                def _err():
                    yield msg
                return _err()
            return msg
        except requests.Timeout:
            msg = "Ollama n'a pas répondu à temps. Réessayez."
            if stream:
                def _err():
                    yield msg
                return _err()
            return msg
        except requests.HTTPError as e:
            msg = f"Erreur Ollama : {e}"
            if stream:
                def _err():
                    yield msg
                return _err()
            return msg

        if not stream:
            data = reponse.json()
            return data.get("response", "")

        def _stream_tokens():
            for ligne in reponse.iter_lines():
                if ligne:
                    donnees = json.loads(ligne)
                    token = donnees.get("response", "")
                    if token:
                        yield token
                    if donnees.get("done", False):
                        break

        return _stream_tokens()
```

---

### `/home/lucmc94/chatbot-local/backend/core/document_manager.py`

```python
"""
core/document_manager.py — Indexation incrémentale des documents.

Tracking via metadata.json par collection (hash SHA256, date, chunk_ids).
Les fichiers metadata sont stockés dans METADATA_DIR (variable d'env).
"""

import hashlib
import json
import logging
import os
import re
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.collection_manager import CollectionManager
from core.parsers import parser_document

logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = int(os.environ.get("CHUNK_SIZE_TOKENS", "450"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "67"))
MAX_CHUNK_TOKENS = 490

_EMBED_HF_MODEL = os.environ.get("EMBED_HF_MODEL", "mixedbread-ai/mxbai-embed-large-v1")


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Charge le tokenizer de l'embedding model. Fallback caractères si indisponible."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(_EMBED_HF_MODEL)
        logger.info(f"Tokenizer chargé : {_EMBED_HF_MODEL}")
        return tok
    except Exception as e:
        logger.warning(f"Tokenizer indisponible ({e}) — fallback 1 token ≈ 3 chars")
        return None


def _count_tokens(text: str) -> int:
    """Compte les tokens du texte selon le tokenizer de l'embedding model."""
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return len(text) // 3

USE_SEMANTIC_CHUNKING = os.environ.get("USE_SEMANTIC_CHUNKING", "false").lower() == "true"
SEMANTIC_BREAKPOINT_THRESHOLD = int(os.environ.get("SEMANTIC_BREAKPOINT_THRESHOLD", "95"))

METADATA_DIR = Path(os.environ.get("METADATA_DIR", "./documents_metadata"))

USE_CONTEXTUAL_RETRIEVAL = os.environ.get("USE_CONTEXTUAL_RETRIEVAL", "false").lower() == "true"
CONTEXTUAL_MAX_WORKERS = int(os.environ.get("CONTEXTUAL_MAX_WORKERS", "3"))
_CONTEXTUAL_PROMPT = (
    "Tu es un assistant technique. Voici un document :\n"
    "<document>\n{document}\n</document>\n\n"
    "Voici un extrait de ce document :\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Génère en 1-2 phrases le contexte de cet extrait : quelle machine est concernée, "
    "quel sujet ou quelle section. Réponds uniquement avec ce contexte, sans introduction."
)


def _enrichir_chunk_contexte(chunk: str, document_complet: str, ollama_url: str, model: str) -> str:
    prompt = _CONTEXTUAL_PROMPT.format(document=document_complet[:6000], chunk=chunk)
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=60,
        )
        resp.raise_for_status()
        contexte = resp.json().get("response", "").strip()
        if contexte:
            return f"[{contexte}]\n\n{chunk}"
    except Exception as e:
        logger.warning(f"Contextual retrieval chunk échoué : {e}")
    return chunk

_MACHINES = ["GEMINI", "SOLO", "COMPAQT", "HYMANCO"]
_MOTS_TYPE_DOC = {"devis", "quotation", "quote", "offre", "dossier", "rfi"}


def _normaliser(texte: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texte.upper())
        if unicodedata.category(c) != "Mn"
    )


def _extraire_metadata_fichier(nom_fichier: str) -> dict:
    stem = Path(nom_fichier).stem
    stem_norm = _normaliser(stem)
    stem_low = stem.lower()

    machine = next((m for m in _MACHINES if m in stem_norm), "")

    type_doc = ""
    if "devis" in stem_low or "quotation" in stem_low or "quote" in stem_low:
        type_doc = "devis"
    elif "offre" in stem_low:
        type_doc = "offre"
    elif "dossier" in stem_low and "technique" in stem_low:
        type_doc = "dossier_technique"
    elif "rfi" in stem_low:
        type_doc = "rfi"

    ref = ""
    m_ref = re.search(r"(AP\d+|\b\d{4,6}\b)", stem)
    if m_ref:
        ref = m_ref.group(0)

    client = ""
    m1 = re.match(r"^(?:AP)?\d+\s*-+\s*([A-Za-z][A-Za-z0-9]+)\s*-", stem)
    if m1:
        candidate = m1.group(1).strip()
        if len(candidate) >= 3 and candidate.lower() not in _MOTS_TYPE_DOC:
            client = candidate

    if not client:
        m2 = re.match(r"^([A-Za-z][A-Za-z]*(?:_[A-Za-z]+)+)\s*[-_ ]+(?:AP|\d)", stem)
        if m2:
            client = m2.group(1).replace("_", " ").strip()

    return {
        "machine": machine,
        "type_doc": type_doc,
        "ref_projet": ref,
        "client": client,
    }


class DocumentManager:
    """Gère l'indexation incrémentale des documents dans les collections."""

    def __init__(self, collection_manager: CollectionManager | None = None):
        self.cm = collection_manager or CollectionManager()

        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=_count_tokens,
        )

        if USE_SEMANTIC_CHUNKING:
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from core.embeddings import get_embeddings
                self.splitter = SemanticChunker(
                    get_embeddings(),
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD,
                )
            except Exception as e:
                logger.warning(f"SemanticChunker indisponible ({e}) — fallback RecursiveCharacterTextSplitter")
                self.splitter = self._recursive_splitter
        else:
            self.splitter = self._recursive_splitter

    def _metadata_path(self, nom_collection: str) -> Path:
        return METADATA_DIR / f"{nom_collection}.json"

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
        chemin = Path(chemin)
        metadata = self._charger_metadata(nom_collection)
        doc_info = metadata["documents"].get(chemin.name)
        if not doc_info:
            return False
        return doc_info["sha256"] == self._calculer_hash(chemin)

    def ajouter_document(self, nom_collection: str, chemin: Path, force: bool = False) -> dict:
        chemin = Path(chemin)

        if not force and self.document_est_indexe(nom_collection, chemin):
            return {
                "status": "skipped",
                "chunks": 0,
                "message": f"{chemin.name} : déjà indexé (hash identique)",
            }

        pages = parser_document(chemin)
        if not pages:
            return {
                "status": "skipped",
                "chunks": 0,
                "message": f"{chemin.name} : aucun texte extrait",
            }

        textes = []
        metadonnees = []
        chunk_ids = []

        meta_fichier = {
            k: v for k, v in _extraire_metadata_fichier(chemin.name).items() if v
        }

        document_complet = ""
        ollama_url = ""
        ollama_model = ""
        if USE_CONTEXTUAL_RETRIEVAL:
            from core.embeddings import OLLAMA_MODEL
            ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
            ollama_model = OLLAMA_MODEL
            document_complet = "\n\n".join(p.texte for p in pages)

        morceaux_par_page: list[tuple[object, list[str]]] = []
        for page in pages:
            morceaux = self.splitter.split_text(page.texte)
            morceaux_proteges = []
            for m in morceaux:
                if _count_tokens(m) > MAX_CHUNK_TOKENS:
                    morceaux_proteges.extend(self._recursive_splitter.split_text(m))
                else:
                    morceaux_proteges.append(m)
            morceaux = morceaux_proteges
            morceaux_par_page.append((page, morceaux))

        if USE_CONTEXTUAL_RETRIEVAL and document_complet:
            tous_morceaux = [(page, m) for page, morceaux in morceaux_par_page for m in morceaux]
            nb_total = len(tous_morceaux)
            enrichis: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=CONTEXTUAL_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _enrichir_chunk_contexte, morceau, document_complet, ollama_url, ollama_model
                    ): idx
                    for idx, (_, morceau) in enumerate(tous_morceaux)
                }
                for future in as_completed(futures):
                    enrichis[futures[future]] = future.result()

            idx = 0
            morceaux_par_page_enrichis: list[tuple[object, list[str]]] = []
            for page, morceaux in morceaux_par_page:
                enrichis_page = [enrichis[idx + i] for i in range(len(morceaux))]
                morceaux_par_page_enrichis.append((page, enrichis_page))
                idx += len(morceaux)
            morceaux_par_page = morceaux_par_page_enrichis

        for page, morceaux in morceaux_par_page:
            for morceau in morceaux:
                cid = str(uuid.uuid4())
                chunk_ids.append(cid)
                textes.append(morceau)
                metadonnees.append({
                    "source": page.source,
                    "page":   page.page,
                    **meta_fichier,
                })

        # Supprimer les anciens chunks si re-indexation
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
        metadata = self._charger_metadata(nom_collection)
        doc_info = metadata["documents"].get(nom_fichier)
        if not doc_info:
            return False

        if doc_info.get("chunk_ids"):
            try:
                db = self.cm.get_collection(nom_collection)
                db.delete(ids=doc_info["chunk_ids"])
            except Exception:
                pass

        del metadata["documents"][nom_fichier]
        self._sauvegarder_metadata(nom_collection, metadata)
        return True

    def lister_documents(self, nom_collection: str) -> list[dict]:
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
```

---

### `/home/lucmc94/chatbot-local/backend/core/parsers.py`

```python
"""
core/parsers.py — Parsers multi-format : PDF, DOCX, TXT/MD, CSV, Excel.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

USE_DOCLING = os.environ.get("USE_DOCLING", "true").lower() == "true"
DOCLING_OCR = os.environ.get("DOCLING_OCR", "false").lower() == "true"
DOCLING_TABLE_MODE = os.environ.get("DOCLING_TABLE_MODE", "fast")

_docling_converter = None


@dataclass
class ParsedPage:
    texte: str
    source: str
    page: int


_EXTENSIONS = {
    ".pdf": "_parser_pdf",
    ".docx": "_parser_docx",
    ".txt": "_parser_texte",
    ".md": "_parser_texte",
    ".csv": "_parser_csv",
    ".xlsx": "_parser_excel",
    ".xls": "_parser_excel",
}


def extensions_supportees() -> list[str]:
    return list(_EXTENSIONS.keys())


def parser_document(chemin: Path) -> list[ParsedPage]:
    chemin = Path(chemin)
    ext = chemin.suffix.lower()
    if ext not in _EXTENSIONS:
        raise ValueError(
            f"Format non supporté : {ext}. "
            f"Formats acceptés : {', '.join(extensions_supportees())}"
        )
    parser_fn = globals()[_EXTENSIONS[ext]]
    return parser_fn(chemin)


def _get_docling_converter():
    global _docling_converter
    if _docling_converter is not None:
        return _docling_converter

    from docling.document_converter import (
        DocumentConverter, PdfFormatOption, WordFormatOption,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, TableFormerMode, TableStructureOptions,
    )
    from docling.pipeline.simple_pipeline import SimplePipeline

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = DOCLING_OCR
    pdf_options.do_table_structure = True
    pdf_options.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.FAST if DOCLING_TABLE_MODE == "fast" else TableFormerMode.ACCURATE,
        do_cell_matching=True,
    )

    allowed_formats = [InputFormat.PDF, InputFormat.DOCX]
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
    }
    try:
        if hasattr(InputFormat, "XLSX"):
            allowed_formats.append(InputFormat.XLSX)
    except Exception:
        pass

    _docling_converter = DocumentConverter(
        allowed_formats=allowed_formats,
        format_options=format_options,
    )
    return _docling_converter


def _docling_result_to_pages(result, chemin: Path) -> list[ParsedPage]:
    from collections import defaultdict
    from docling_core.types.doc import TableItem, TextItem

    doc = result.document
    pages_content: dict[int, list[str]] = defaultdict(list)

    for item, _level in doc.iterate_items():
        text = None
        if isinstance(item, TableItem):
            try:
                df = item.export_to_dataframe(doc=doc)
                text = df.to_markdown(index=False)
            except Exception:
                try:
                    text = item.export_to_html(doc=doc)
                except Exception:
                    pass
        elif isinstance(item, TextItem):
            text = item.text

        if not text or not text.strip():
            continue

        page_no = 1
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no

        pages_content[page_no].append(text.strip())

    if not pages_content:
        markdown = doc.export_to_markdown()
        if markdown.strip():
            return [ParsedPage(texte=markdown.strip(), source=chemin.name, page=1)]
        return []

    return [
        ParsedPage(
            texte="\n\n".join(texts),
            source=chemin.name,
            page=page_no,
        )
        for page_no, texts in sorted(pages_content.items())
        if texts and any(t.strip() for t in texts)
    ]


def _parser_pdf(chemin: Path) -> list[ParsedPage]:
    if USE_DOCLING:
        try:
            converter = _get_docling_converter()
            result = converter.convert(str(chemin))
            pages = _docling_result_to_pages(result, chemin)
            if pages:
                return pages
        except Exception as e:
            logger.warning(f"Docling échoué pour {chemin.name} : {e} — fallback PyMuPDF4LLM")
    return _parser_pdf_pymupdf(chemin)


def _parser_pdf_pymupdf(chemin: Path) -> list[ParsedPage]:
    import pymupdf4llm
    pages = []
    try:
        pages_md = pymupdf4llm.to_markdown(str(chemin), page_chunks=True)
    except Exception as e:
        print(f"  Impossible de lire {chemin.name} : {e}")
        return pages
    for page_data in pages_md:
        texte = page_data.get("text", "")
        num_page = page_data.get("metadata", {}).get("page", 1)
        num_page = num_page + 1 if isinstance(num_page, int) else 1
        if texte.strip():
            pages.append(ParsedPage(texte=texte.strip(), source=chemin.name, page=num_page))
    return pages


def _parser_docx(chemin: Path) -> list[ParsedPage]:
    if USE_DOCLING:
        try:
            converter = _get_docling_converter()
            result = converter.convert(str(chemin))
            pages = _docling_result_to_pages(result, chemin)
            if pages:
                return pages
        except Exception as e:
            logger.warning(f"Docling échoué pour {chemin.name} : {e} — fallback python-docx")
    return _parser_docx_legacy(chemin)


def _parser_docx_legacy(chemin: Path) -> list[ParsedPage]:
    from docx import Document
    doc = Document(str(chemin))
    texte_complet = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not texte_complet.strip():
        return []
    return [ParsedPage(texte=texte_complet.strip(), source=chemin.name, page=1)]


def _parser_texte(chemin: Path) -> list[ParsedPage]:
    texte = chemin.read_text(encoding="utf-8", errors="ignore")
    if not texte.strip():
        return []
    return [ParsedPage(texte=texte.strip(), source=chemin.name, page=1)]


def _parser_csv(chemin: Path) -> list[ParsedPage]:
    import pandas as pd
    df = pd.read_csv(str(chemin))
    texte = df.to_string(index=False)
    if not texte.strip():
        return []
    return [ParsedPage(texte=texte.strip(), source=chemin.name, page=1)]


def _parser_excel(chemin: Path) -> list[ParsedPage]:
    if USE_DOCLING and chemin.suffix.lower() == ".xlsx":
        try:
            from docling.datamodel.base_models import InputFormat
            if hasattr(InputFormat, "XLSX"):
                converter = _get_docling_converter()
                result = converter.convert(str(chemin))
                pages = _docling_result_to_pages(result, chemin)
                if pages:
                    return pages
        except Exception as e:
            logger.warning(f"Docling XLSX échoué ({e}) — fallback pandas ({chemin.name})")
    return _parser_excel_pandas(chemin)


def _parser_excel_pandas(chemin: Path) -> list[ParsedPage]:
    import pandas as pd
    ROWS_PER_BLOCK = 30
    pages = []
    try:
        excel_file = pd.ExcelFile(str(chemin))
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            if df.empty:
                continue
            for i in range(0, len(df), ROWS_PER_BLOCK):
                bloc = df.iloc[i: i + ROWS_PER_BLOCK]
                texte = f"# Feuille: {sheet_name} (lignes {i + 1}-{i + len(bloc)})\n\n"
                texte += bloc.to_markdown(index=False)
                pages.append(ParsedPage(
                    texte=texte.strip(),
                    source=f"{chemin.name} (Feuille: {sheet_name})",
                    page=len(pages) + 1,
                ))
    except Exception as e:
        logger.warning(f"Impossible de lire {chemin.name} : {e}")
        return []
    return pages
```

---

### `/home/lucmc94/chatbot-local/backend/api/routes/chat.py`

```python
"""Chat API routes with SSE streaming."""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.domain.models.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])

MAX_HISTORY_MESSAGES = 10


def _format_history(history: list) -> str:
    """Format conversation history for the prompt."""
    if not history:
        return ""
    formatted = []
    recent_history = history[-MAX_HISTORY_MESSAGES:]
    for msg in recent_history:
        role = "User" if msg.role == "user" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


async def _stream_rag_response(
    message: str, collection_name: str, prompt_name: str, history: list
) -> AsyncGenerator[str, None]:
    """Stream RAG response as SSE events."""
    from core.collection_manager import CollectionManager
    from core.search import RAGEngine

    try:
        cm = CollectionManager()
        if not cm.collection_existe(collection_name):
            yield f"data: {json.dumps({'error': f'Collection {collection_name} not found'})}\n\n"
            return

        rag = RAGEngine(collection_name, prompt_name=prompt_name, collection_manager=cm)

        history_text = _format_history(history)
        result = rag.generer_avec_sources(message, stream=True, history=history_text)

        # Stream tokens
        for token in result["reponse"]:
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Send sources at the end
        yield f"data: {json.dumps({'sources': result['sources'], 'done': True})}\n\n"

    except ValueError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Internal error: {str(e)}'})}\n\n"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_rag_response(
            request.message,
            request.collection_name,
            request.prompt_name,
            request.history
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest) -> ChatResponse:
    from core.collection_manager import CollectionManager
    from core.search import RAGEngine

    cm = CollectionManager()
    if not cm.collection_existe(request.collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found")

    try:
        rag = RAGEngine(request.collection_name, prompt_name=request.prompt_name, collection_manager=cm)
        history_text = _format_history(request.history)
        result = rag.generer_avec_sources(request.message, stream=False, history=history_text)
        return ChatResponse(response=result["reponse"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### `/home/lucmc94/chatbot-local/backend/api/routes/documents.py`

```python
"""Documents management API routes."""

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.api.dependencies import get_settings

router = APIRouter(prefix="/api/collections/{collection_name}/documents", tags=["documents"])

settings = get_settings()


class DocumentInfo(BaseModel):
    nom: str
    date: str
    nb_chunks: int
    nb_pages: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class IndexResult(BaseModel):
    status: str
    chunks: int
    message: str


@router.get("", response_model=DocumentListResponse)
async def list_documents(collection_name: str) -> DocumentListResponse:
    from core.collection_manager import CollectionManager
    from core.document_manager import DocumentManager

    cm = CollectionManager()
    if not cm.collection_existe(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    dm = DocumentManager(cm)
    docs = dm.lister_documents(collection_name)
    return DocumentListResponse(documents=[DocumentInfo(**d) for d in docs])


@router.post("", response_model=IndexResult, status_code=201)
async def upload_document(
    collection_name: str,
    file: UploadFile = File(...),
    force: bool = Query(False, description="Force re-indexation even if document exists"),
) -> IndexResult:
    from core.collection_manager import CollectionManager
    from core.document_manager import DocumentManager

    cm = CollectionManager()

    if not cm.collection_existe(collection_name):
        cm.creer_collection(collection_name)

    suffix = Path(file.filename or "document").suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        final_path = tmp_path.parent / (file.filename or "document")
        tmp_path.rename(final_path)

        dm = DocumentManager(cm)
        result = dm.ajouter_document(collection_name, final_path, force=force)
        return IndexResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        for p in [tmp_path, final_path]:
            if p.exists():
                p.unlink()


@router.delete("/{document_name}", status_code=204)
async def delete_document(collection_name: str, document_name: str) -> None:
    from core.collection_manager import CollectionManager
    from core.document_manager import DocumentManager

    cm = CollectionManager()
    if not cm.collection_existe(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    dm = DocumentManager(cm)
    if not dm.supprimer_document(collection_name, document_name):
        raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
```

---

### `/home/lucmc94/chatbot-local/backend/api/routes/health.py`

```python
"""Health check endpoint."""

import subprocess

import chromadb
import httpx
from fastapi import APIRouter, Depends

from backend.api.dependencies import get_settings
from backend.config import Settings
from backend.domain.models import ApiResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


async def check_ollama(settings: Settings) -> str:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            if response.status_code == 200:
                return "ok"
            return "unavailable"
    except Exception:
        return "unavailable"


async def check_chromadb(settings: Settings) -> str:
    try:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        client.heartbeat()
        return "ok"
    except Exception:
        return "unavailable"


def check_gpu() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "detected"
        return "not_detected"
    except Exception:
        return "not_detected"


@router.get("/status")
async def llm_status(settings: Settings = Depends(get_settings)) -> dict:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_url}/api/ps")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_base = settings.ollama_model.split(":")[0]
                ready = any(model_base in m.get("name", "") for m in models)
                return {"llm_ready": ready}
    except Exception:
        pass
    return {"llm_ready": False}


@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)) -> ApiResponse:
    ollama_status = await check_ollama(settings)
    chromadb_status = await check_chromadb(settings)
    gpu_status = check_gpu()

    data = {
        "ollama": ollama_status,
        "chromadb": chromadb_status,
        "gpu": gpu_status,
    }

    degraded = []
    if ollama_status == "unavailable":
        degraded.append("Ollama unreachable")
    if chromadb_status == "unavailable":
        degraded.append("ChromaDB unreachable")

    message = f"Degraded: {', '.join(degraded)}" if degraded else None
    return ApiResponse.success(data=data, message=message)
```

---

### `/home/lucmc94/chatbot-local/backend/api/routes/collections.py`

```python
"""Collections management API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/collections", tags=["collections"])


class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")


class CollectionInfo(BaseModel):
    name: str
    document_count: int


class CollectionListResponse(BaseModel):
    collections: list[str]


@router.get("", response_model=CollectionListResponse)
async def list_collections() -> CollectionListResponse:
    from core.collection_manager import CollectionManager
    cm = CollectionManager()
    return CollectionListResponse(collections=cm.lister_collections())


@router.post("", response_model=CollectionInfo, status_code=201)
async def create_collection(request: CollectionCreate) -> CollectionInfo:
    from core.collection_manager import CollectionManager
    cm = CollectionManager()
    if cm.collection_existe(request.name):
        raise HTTPException(status_code=409, detail=f"Collection '{request.name}' already exists")
    cm.creer_collection(request.name)
    return CollectionInfo(name=request.name, document_count=0)


@router.get("/{name}", response_model=CollectionInfo)
async def get_collection(name: str) -> CollectionInfo:
    from core.collection_manager import CollectionManager
    cm = CollectionManager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    db = cm.get_collection(name)
    try:
        count = db._collection.count()
    except Exception:
        count = 0
    return CollectionInfo(name=name, document_count=count)


@router.delete("/{name}", status_code=204)
async def delete_collection(name: str) -> None:
    from core.collection_manager import CollectionManager
    cm = CollectionManager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    cm.supprimer_collection(name)
```

---

### `/home/lucmc94/chatbot-local/backend/main.py`

```python
"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.dependencies import get_settings
from backend.api.routes import chat_router, collections_router, documents_router, health_router

logger = logging.getLogger(__name__)

settings = get_settings()


def _warmup_ollama() -> None:
    from core.embeddings import OLLAMA_API_GENERATE, OLLAMA_BASE_URL, EMBEDDING_MODEL

    # 1. Warm-up embed
    try:
        requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": "warmup"},
            timeout=60,
        )
        logger.info("Warmup embed OK")
    except Exception as e:
        logger.warning(f"Warmup embed échoué : {e}")

    # 2. Warm-up LLM : timeout=None intentionnellement (chargement peut dépasser 5 min)
    try:
        requests.post(
            OLLAMA_API_GENERATE,
            json={"model": "llama3.1:8b", "prompt": "warmup", "stream": False,
                  "options": {"num_predict": 1}},
            timeout=None,
        )
        logger.info("Warmup LLM OK")
    except Exception as e:
        logger.warning(f"Warmup LLM échoué : {e}")


def _warmup_reranker() -> None:
    try:
        from core.search import _get_reranker, _get_colbert
        _get_colbert() or _get_reranker()
        logger.info("Warmup reranker OK")
    except Exception as e:
        logger.warning(f"Warmup reranker échoué : {e}")


def _warmup_all() -> None:
    """Séquence de warmup dans un seul thread pour éviter le deadlock ModuleLock."""
    _warmup_ollama()
    _warmup_reranker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lance le pre-warm Ollama + reranker en arrière-plan au démarrage."""
    loop = asyncio.get_event_loop()
    logger.info("Démarrage pre-warm modèles (arrière-plan)…")
    loop.run_in_executor(None, _warmup_all)
    yield


app = FastAPI(
    title="chatbot-local",
    description="RAG chatbot for VLM Robotics product knowledge",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(collections_router)
app.include_router(documents_router)


@app.get("/")
async def root():
    return {"message": "chatbot-local API", "version": "0.1.0"}
```

---

### `/home/lucmc94/chatbot-local/backend/domain/models/chat.py`

```python
"""Chat domain models."""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    collection_name: str = Field(..., min_length=1, description="ChromaDB collection to search")
    prompt_name: str = Field(default="defaut", description="Prompt template name")
    history: list[ChatMessage] = Field(default=[], description="Previous messages for context")


class ChatSource(BaseModel):
    fichier: str
    page: str | int
    score: float


class ChatResponse(BaseModel):
    response: str
    sources: list[ChatSource]
```

---

### `/home/lucmc94/chatbot-local/backend/api/dependencies.py`

```python
"""Dependency injection wiring for FastAPI."""

from functools import lru_cache

from backend.config.settings import Settings


@lru_cache
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()


# Port implementations will be registered here as adapters are implemented
# Example:
# def get_llm_port() -> LlmPort:
#     settings = get_settings()
#     return OllamaAdapter(settings.ollama_url)
```

---

### `/home/lucmc94/chatbot-local/backend/config/settings.py`

```python
"""Application settings."""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Ollama settings
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

    # ChromaDB settings
    chroma_host: str = "chromadb"
    chroma_port: int = 8100

    # Document storage
    documents_path: str = "/app/documents"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://frontend:3000"]
```

---

### `/home/lucmc94/chatbot-local/docker-compose.yml`

```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
    networks:
      - chatbot-network

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_URL=http://ollama:11434
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - METADATA_DIR=/app/documents/metadata
      - CORS_ORIGINS=["http://localhost:3000","http://frontend:3000"]
      - OLLAMA_EMBED_MODEL=mxbai-embed-large
      - EMBED_HF_MODEL=mixedbread-ai/mxbai-embed-large-v1
      - TOKENIZERS_PARALLELISM=false
      - HF_HUB_OFFLINE=1
      - USE_DOCLING=true
      - DOCLING_OCR=false
      - DOCLING_TABLE_MODE=accurate
      - USE_RERANKER=true
      - RERANKER_MODEL=BAAI/bge-reranker-v2-m3
      - USE_HYBRID_SEARCH=true
      - USE_CONTEXTUAL_RETRIEVAL=false
      - CONTEXTUAL_MAX_WORKERS=3
      - USE_SEMANTIC_CHUNKING=true
      - SEMANTIC_BREAKPOINT_THRESHOLD=95
      - USE_COLBERT=false
      - COLBERT_MODEL=colbert-ir/colbertv2.0
      - USE_HF_EMBEDDINGS=true
    volumes:
      - documents_store:/app/documents
      - hf_cache:/root/.cache/huggingface
    depends_on:
      - ollama
      - chromadb
    networks:
      - chatbot-network

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    environment:
      - OLLAMA_LOAD_TIMEOUT=900
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - chatbot-network

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8100:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - ANONYMIZED_TELEMETRY=False
      - ALLOW_RESET=True
    networks:
      - chatbot-network

networks:
  chatbot-network:
    driver: bridge

volumes:
  chroma_data:
  documents_store:
  ollama_models:
  hf_cache:
```

---

### `/home/lucmc94/chatbot-local/backend/Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
# Backend Dockerfile - Multi-stage build
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-ml.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements-ml.txt

COPY backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

FROM python:3.12-slim AS production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY backend/ ./backend/

RUN mkdir -p /app/documents

# backend/core/ prend priorité sur core/ racine via PYTHONPATH
ENV PYTHONPATH=/app/backend:/app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### `/home/lucmc94/chatbot-local/frontend/app/components/Chat.tsx`

```typescript
"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import { streamChat, fetchCollections, fetchLLMStatus } from "../lib/api";
import type { ChatMessage, ChatSource } from "../lib/types";

function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

// ... (ChatInput, Sources, MessageBubble components)

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [collection, setCollection] = useState("");
  const [collections, setCollections] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [llmReady, setLlmReady] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch collections on mount
  useEffect(() => {
    fetchCollections().then((cols) => {
      setCollections(cols);
      if (cols.length > 0 && !collection) setCollection(cols[0]);
    });
  }, []);

  // Poll LLM status every 3s until ready (then stops)
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    const check = async () => {
      const ready = await fetchLLMStatus();
      if (ready) { setLlmReady(true); clearInterval(interval); }
    };
    check();
    interval = setInterval(check, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleSend = async (content: string) => {
    setError(null);
    const userMessage: ChatMessage = { id: generateId(), role: "user", content };
    const assistantMessage: ChatMessage = { id: generateId(), role: "assistant", content: "", isStreaming: true };

    const history = messages
      .filter((m) => m.content && !m.isStreaming)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    try {
      // NOTE: promptName = collection (same variable used for both)
      for await (const event of streamChat(content, collection, collection, history)) {
        if (event.error) {
          setError(event.error);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, content: `Erreur: ${event.error}`, isStreaming: false }
                : m
            )
          );
          break;
        }
        if (event.token) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id ? { ...m, content: m.content + event.token } : m
            )
          );
        }
        if (event.done && event.sources) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, sources: event.sources, isStreaming: false }
                : m
            )
          );
        }
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Erreur de connexion";
      setError(errorMsg);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessage.id
            ? { ...m, content: `Erreur: ${errorMsg}`, isStreaming: false }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      {/* header, messages list, LLM loading banner, ChatInput */}
      {!llmReady && (
        <div className="...amber banner...">
          Modèle IA en cours de chargement... (peut prendre 1-2 min au démarrage)
        </div>
      )}
      {/* ChatInput disabled={isStreaming || !llmReady} */}
    </div>
  );
}
```

---

### `/home/lucmc94/chatbot-local/frontend/app/lib/api.ts`

```typescript
import type { SSEEvent } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export async function* streamChat(
  message: string,
  collectionName: string,
  promptName: string = "defaut",
  history: HistoryMessage[] = []
): AsyncGenerator<SSEEvent> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      collection_name: collectionName,
      prompt_name: promptName,
      history,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6)) as SSEEvent;
          yield data;
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}

export async function fetchLLMStatus(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/api/v1/status`);
    if (!response.ok) return false;
    const data = await response.json();
    return data.llm_ready === true;
  } catch {
    return false;
  }
}

export async function fetchCollections(): Promise<string[]> {
  const response = await fetch(`${API_URL}/api/collections`);
  if (!response.ok) return [];
  const data = await response.json();
  return data.collections || [];
}
```

---

### `/home/lucmc94/chatbot-local/frontend/app/admin/page.tsx`

```typescript
"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Upload documents (multi-fichiers, séquentiel)
// handleUpload : pour chaque fichier → POST /api/collections/{name}/documents
// Progress bar : current/total affiché
// handleDeleteDocument, handleCreateCollection, handleDeleteCollection : fetch calls

// NOTE : CORS_ORIGINS dans docker-compose est une string JSON brute,
// non gérée nativement par pydantic-settings → risque de désérialisation incorrecte
```

---

## QUESTIONS D'AUDIT

---

## AXE 1 — Architecture & maintenabilité

**Q1.1 — Robustesse des imports PYTHONPATH**

La résolution des imports `from core.xxx` dépend de `PYTHONPATH=/app/backend:/app` dans le Dockerfile. Si ce PYTHONPATH n'est pas défini (dev local, tests CI, autre Dockerfile), tous les imports core échouent silencieusement.

Questions :
- Est-ce une approche robuste pour un projet maintenable à long terme ? Quelles alternatives ? (`from backend.core.xxx` partout, ou un vrai package installable via `pip install -e .`)
- Y a-t-il un risque concret de confusion entre le `core/` racine (si présent) et `backend/core/` (actif) selon l'ordre du PYTHONPATH ?
- Comment sécuriser cette résolution pour les tests locaux sans Docker ?

**Q1.2 — Respect réel de l'architecture hexagonale**

Les routes (`chat.py`, `collections.py`, `documents.py`) importent directement `from core.collection_manager import CollectionManager` et `from core.search import RAGEngine` en lazy import dans les handlers. Les `domain/ports/` définissent des interfaces (LLMPort, EmbeddingPort, VectorStorePort) mais aucun adapter ne les implémente (commentaire dans `dependencies.py`).

Questions :
- Les routes ont-elles un couplage fort avec les modules core ? Quel est l'impact sur la testabilité ?
- Les ports/adapters sont-ils de la dette technique à prioriser ou une over-engineering pour ce POC ?
- Comment migrer proprement vers les adapters sans réécriture massive ?

**Q1.3 — Imports dynamiques dans les handlers**

Chaque handler FastAPI importe `CollectionManager`, `DocumentManager`, `RAGEngine` à l'intérieur de la fonction (lazy import), donc à chaque requête HTTP. Exemple dans `chat.py:37-38` :
```python
from core.collection_manager import CollectionManager
from core.search import RAGEngine
```

Questions :
- En Python, les imports sont mis en cache dans `sys.modules` après le premier import. Ces imports répétés ont-ils un coût réel de performance ? Sont-ils simplement un lookup dict ?
- Ces imports dynamiques avaient-ils un but précis (éviter des circular imports au moment du démarrage) ? L'intention est-elle claire ?
- La lisibilité en souffre-t-elle ? Pattern recommandé pour FastAPI ?

**Q1.4 — Cohérence des responsabilités : historique**

`_format_history()` existe dans `chat.py` (lignes 17-30) ET la gestion de l'historique (session scoping, query rewriting) est aussi dans `search.py`.

Questions :
- Y a-t-il duplication de logique ou séparation justifiée (format présentation vs enrichissement retrieval) ?
- `_booster_sources_session` dans `search.py` est déclaré mais jamais appelé dans le pipeline `rechercher()` — est-ce une fonctionnalité orpheline ?

**Q1.5 — Blocking I/O dans handlers async**

`_stream_rag_response` dans `chat.py` est une coroutine `async`, mais elle appelle `rag.generer_avec_sources()` qui fait du blocking I/O :
- `requests.post()` vers Ollama (bloquant)
- `self.db.similarity_search_with_score()` vers ChromaDB HTTP (bloquant)
- Le reranker cross-encoder BGE (CPU, bloquant)

Question : Quel est l'impact réel sur uvicorn avec un seul worker ? Avec plusieurs workers ? Comment corriger proprement (asyncio.to_thread, httpx async, etc.) ?

---

## AXE 2 — Performance & retrieval

**Q2.1 — `get_embeddings()` appelé à chaque requête**

Dans `collection_manager.py`, `creer_collection()` et `get_collection()` appellent `get_embeddings()` à chaque invocation, ce qui instancie `HFEmbeddings()` (et donc charge `HuggingFaceEmbeddings`) à chaque requête chat ET à chaque ingest.

```python
def get_collection(self, nom: str) -> Chroma:
    return Chroma(
        client=self._client,
        collection_name=nom,
        embedding_function=get_embeddings(),  # ← instanciation à chaque appel
    )
```

Questions :
- Quel est le coût réel d'instancier `HuggingFaceEmbeddings` à chaque fois (chargement modèle en mémoire ou juste création objet Python) ?
- Faut-il un singleton ou `lru_cache` sur `get_embeddings()` ? Quels risks de thread-safety ?
- `_stream_rag_response` crée de nouveaux `CollectionManager` et `RAGEngine` à chaque requête chat — est-ce le pattern optimal ?

**Q2.2 — Thread-safety des singletons globaux**

`_reranker_instance`, `_colbert_instance`, `_docling_converter`, `_bm25_cache` sont des variables globales mutables. En Python, le GIL protège les opérations bytecode élémentaires, mais pas les blocs critiques plus larges.

Questions :
- Le pattern "check-then-set" dans `_get_reranker()` (lignes 392-412) est-il thread-safe sous uvicorn multi-workers ?
- Sous uvicorn multi-workers (mode `--workers N`), chaque worker est un process distinct → pas de mémoire partagée → les singletons sont-ils re-créés par worker ?
- `_bm25_cache` est un dict Python global : les opérations dict en Python sont-elles atomiques ? Y a-t-il un risque de corruption en cas de write concurrent ?

**Q2.3 — BM25 et race condition pendant ingest**

`_get_or_build_bm25()` invalide le cache quand `nb_chunks` change. Si un ingest est en cours (nouveaux chunks en cours d'ajout) pendant qu'une requête chat reconstruit l'index BM25, l'index peut être partiellement stale.

Questions :
- Ce risque est-il réel dans ce contexte mono-utilisateur POC ? Et en production multi-utilisateurs ?
- Solution recommandée : verrou (`threading.Lock`), invalidation après commit, ou accepter le staleness ?

**Q2.4 — `_adapter_parametres()` : `db._collection.count()` à chaque requête**

```python
def _adapter_parametres(self) -> tuple[int, int]:
    try:
        nb_chunks = self.db._collection.count()  # appel HTTP ChromaDB à chaque requête
    except Exception:
        nb_chunks = 0
```

Questions :
- Quel est le coût de ce round-trip HTTP ChromaDB à chaque requête chat ? (~1-5ms en local Docker)
- Ce count est-il utilisé uniquement pour calibrer K et num_ctx, donc une valeur stale de quelques secondes serait acceptable ?
- Pattern recommandé : cache TTL de 60s, ou passer nb_chunks comme paramètre lors de la construction de RAGEngine ?

**Q2.5 — SemanticChunker : N appels embeddings synchrones dans un handler async**

Quand `USE_SEMANTIC_CHUNKING=true`, `DocumentManager.__init__()` crée un `SemanticChunker` qui appelle `get_embeddings()`. Lors de l'ingest, `self.splitter.split_text(page.texte)` fait N appels embeddings bloquants.

Questions :
- `upload_document` est un handler `async`, `DocumentManager.ajouter_document()` est synchrone et bloquant — impact sur la boucle d'événements ?
- Solution : `asyncio.to_thread(dm.ajouter_document, ...)` ? Ou un endpoint dédié non-async ?
- Le SemanticChunker de LangChain est-il bien adapté à un pipeline batch ou prévu pour du one-shot ?

**Q2.6 — Query rewriting sans condition de court-circuit**

`_reformuler_question()` appelle Ollama pour chaque message même si `history` a une seule ligne (un seul échange précédent).

```python
def _reformuler_question(question: str, history: str) -> str:
    if not history:
        return question
    # Appel Ollama à chaque fois si history non vide, même pour 1 seul message
```

Questions :
- Y a-t-il un seuil raisonnable (ex: len(history) < 100 chars → skip rewriting) ?
- Le coût du rewriting (appel Ollama synchrone, 15s timeout) est-il justifié pour chaque requête avec historique minimal ?
- Le `timeout=15` est-il suffisant si Ollama est sous charge (en train de générer en streaming pour un autre utilisateur) ?

---

## AXE 3 — Robustesse & fiabilité

**Q3.1 — Propagation des exceptions dans `_stream_tokens()`**

```python
def _stream_tokens():
    for ligne in reponse.iter_lines():
        if ligne:
            donnees = json.loads(ligne)  # que se passe-t-il si JSON malformé ?
            token = donnees.get("response", "")
            if token:
                yield token
            if donnees.get("done", False):
                break
```

Questions :
- Si la connexion Ollama est coupée en cours de stream, `iter_lines()` lève une `requests.exceptions.ChunkedEncodingError`. Est-elle bien propagée jusqu'au handler SSE dans `chat.py` ?
- Dans `_stream_rag_response`, la boucle `for token in result["reponse"]` n'est pas wrappée dans un try/except — l'exception serait-elle swallowed ou propagée au client SSE ?
- Un `json.loads(ligne)` sur une ligne malformée lève `json.JSONDecodeError` non catchée dans le générateur — impact ?

**Q3.2 — Bug `finally` dans `upload_document`**

```python
try:
    final_path = tmp_path.parent / (file.filename or "document")
    tmp_path.rename(final_path)  # ← tmp_path n'existe plus après rename
    ...
finally:
    for p in [tmp_path, final_path]:  # tmp_path.exists() → False
        if p.exists():
            p.unlink()
```

Après `tmp_path.rename(final_path)`, `tmp_path` n'existe plus. Dans le `finally`, `tmp_path.exists()` retourne `False` (silencieux). Mais `final_path` n'est définie que si l'on sort du bloc `try` après le `rename`. Si le `rename` lui-même échoue (permissions, nom invalide), `final_path` peut être non définie et `NameError` dans le `finally`.

Questions :
- Y a-t-il un `NameError` potentiel sur `final_path` dans le `finally` si le `rename` échoue avant l'affectation ?
- La logique de cleanup est-elle correcte ? Comment la réécrire proprement ?

**Q3.3 — Fingerprint de déduplication `doc.page_content[:150]`**

Utilisé dans `_rrf_fusion()` (`uid = doc.page_content[:150]`) et dans `_keyword_fallback_search()` (`uid = text[:120]`).

Questions :
- Quelle est la probabilité de collision pour des chunks techniques (brochures, offres) ayant les mêmes 150 premiers caractères mais un contenu différent ?
- Une collision provoquerait-elle une perte de chunk silencieuse dans la fusion RRF ou dans la déduplication keyword ? Quel serait l'impact utilisateur ?
- Alternative recommandée : hash SHA256 du contenu complet, ou ID ChromaDB du chunk ?

**Q3.4 — Session scoping + seuil relatif : interaction problématique**

`_booster_sources_session()` multiplie le score reranker par 2.0. Ensuite, dans `rechercher()` :
```python
seuil_relatif = score_max_r * 0.1
resultats = [(doc, s) for doc, s in resultats if s >= seuil_relatif]
```
Si le meilleur chunk a score boosté = 0.9 × 2 = 1.8, le seuil relatif devient 0.18. Des chunks non boostés pertinents à score 0.15 seraient éliminés.

Note : `_booster_sources_session()` est défini mais n'est PAS appelé dans le pipeline actuel de `rechercher()`.

Questions :
- Cette fonction orpheline présente-t-elle un risque si elle est branchée plus tard sans tenir compte de l'interaction avec le seuil relatif ?
- Comment concevoir un boosting compatible avec le seuil relatif (ex: appliquer le boost APRÈS le filtrage par seuil) ?

**Q3.5 — Race condition dans `_charger_metadata` lors d'ingests parallèles**

`ajouter_document()` appelle `_charger_metadata()` deux fois (lignes 250 et 345) sans verrou. Si deux ingests parallèles sur la même collection lisent le même fichier JSON, l'un des deux peut écraser les modifications de l'autre.

```python
# Premier appel (ligne 250)
if not force and self.document_est_indexe(nom_collection, chemin):  # lit le JSON

# ...indexation...

# Deuxième appel (ligne 345)
metadata = self._charger_metadata(nom_collection)  # re-lit le JSON
```

Questions :
- Ce cas est-il réaliste (upload admin multi-fichiers séquentiel en JS → pas de parallélisme côté serveur) ?
- Si plusieurs utilisateurs uploadent en même temps → race condition réelle ?
- Solution : `threading.Lock` par collection, ou base de données metadata à la place de JSON ?

**Q3.6 — `lru_cache(maxsize=1)` sur `_get_tokenizer()` : thread-safety**

`@lru_cache` est thread-safe en Python 3.2+ (implémentation utilise un verrou interne). La question porte sur l'initialisation du tokenizer elle-même.

Questions :
- Si deux threads appellent `_get_tokenizer()` simultanément et que le cache est vide, y a-t-il un double chargement du tokenizer ?
- `lru_cache` garantit-il que la fonction n'est appelée qu'une seule fois même en cas d'appels concurrents ?

**Q3.7 — Frontend bloqué si warmup échoue**

Le frontend poll `GET /api/v1/status` toutes les 3 secondes. Si Ollama crashe définitivement (VRAM insuffisante, modèle corrompu), `llm_ready` reste `false` indéfiniment et le frontend est bloqué sur la bannière "Modèle IA en cours de chargement".

Questions :
- Y a-t-il un timeout ou un nombre maximum de tentatives de polling côté frontend ?
- Comment distinguer "en cours de chargement" de "Ollama en erreur permanente" côté `/api/v1/status` ?
- Solution recommandée : timeout frontend (ex: 5 min) + message d'erreur explicite, ou endpoint de status enrichi avec état du warmup ?

---

## AXE 4 — Qualité du code

**Q4.1 — Variables globales lues à l'import-time**

Dans `embeddings.py` (lignes 25-28) :
```python
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
```

Ces variables sont lues à l'import-time (pas au moment de l'appel). En tests locaux sans variables d'env, les valeurs par défaut sont utilisées (`localhost:11434`). Il existe un double source de vérité : `embeddings.py` lit les env vars directement, et `config/settings.py` les lit via pydantic-settings — mais les deux ne sont pas connectés.

Questions :
- Quel est le risque concret de ce double source de vérité (Settings vs os.environ direct) ?
- En tests unitaires qui patchent les env vars après import, ces constantes globales ne seraient pas mises à jour. Quelle approche recommandez-vous ?
- La valeur `OLLAMA_API_GENERATE` est construite à l'import → non patchable facilement en tests. Est-ce un problème ?

**Q4.2 — `_appeler_ollama` comme staticmethod**

```python
@staticmethod
def _appeler_ollama(prompt: str, stream: bool = True, num_ctx: int = NUM_CTX_MIN):
```

Cette méthode ne dépend d'aucun état d'instance (`self`). Elle utilise `OLLAMA_MODEL` et `OLLAMA_API_GENERATE` qui sont des globals de module.

Questions :
- Est-ce un anti-pattern d'avoir une staticmethod qui pourrait être une fonction module-level ?
- Quel est l'impact sur la testabilité (mock plus difficile sur une staticmethod) ?

**Q4.3 — Création de CollectionManager et RAGEngine à chaque requête**

Dans `_stream_rag_response` (chat.py) :
```python
cm = CollectionManager()  # nouveau client HTTP ChromaDB à chaque requête
rag = RAGEngine(collection_name, prompt_name=prompt_name, collection_manager=cm)
```

Questions :
- Combien de connexions TCP ChromaDB sont créées par requête ? Sont-elles poolées par la librairie chromadb-client ?
- Un pool de connexions ou un singleton `CollectionManager` injecté via FastAPI `Depends()` serait-il préférable ?

**Q4.4 — Warmup LLM avec `timeout=None`**

```python
requests.post(
    OLLAMA_API_GENERATE,
    json={"model": "llama3.1:8b", "prompt": "warmup", "stream": False,
          "options": {"num_predict": 1}},
    timeout=None,  # ← attend indéfiniment
)
```

Si Ollama ne démarre jamais (crash, VRAM insuffisante, modèle manquant), ce thread de warmup reste bloqué indéfiniment dans l'executor.

Questions :
- Ce thread bloqué a-t-il un impact sur les performances de l'application FastAPI (il est dans un ThreadPoolExecutor, pas dans la boucle d'événements) ?
- Quel timeout maximum raisonnable pour le warmup LLM ? Comment gérer proprement le cas Ollama-crashé ?
- Le thread warmup bloqué empêche-t-il le shutdown propre du container Docker ?

**Q4.5 — Memory leaks potentiels**

Questions sur :
- Les connexions `chromadb.HttpClient` créées à chaque requête dans chaque handler — sont-elles proprement fermées ?
- Le générateur `_stream_tokens()` retourné par `_appeler_ollama()` : si le client SSE se déconnecte avant la fin, le générateur est-il GC'd proprement ? La connexion `requests.Response` est-elle fermée ?
- Les `NamedTemporaryFile` dans `upload_document` : si le handler est annulé (client disconnect), le `finally` est-il garanti d'être exécuté ?

**Q4.6 — Domain ports non implémentés**

`domain/ports/` définit des interfaces abstraites (`LLMPort`, `EmbeddingPort`, `VectorStorePort`) mais aucun adapter ne les implémente. Le commentaire dans `dependencies.py` dit "Port implementations will be registered here as adapters are implemented."

Questions :
- Faut-il supprimer ces ports (dette vide) ou les garder comme contrat d'interface pour une future migration ?
- Si on implémente les adapters, quelles seraient les 3 premières à implémenter pour le meilleur ROI ?
- L'absence d'adapters rend-elle les tests unitaires impossibles (pas de mock possible sans refactoring) ?

**Q4.7 — Couverture de tests quasi-nulle**

Questions :
- Quels sont les 5 tests les plus critiques à implémenter en priorité (ratio risque/effort) ?
- Comment mocker Ollama et ChromaDB pour des tests unitaires sans Docker ?
- Y a-t-il des points du code trop couplés pour être testés unitairement en l'état ?

**Q4.8 — Bug potentiel : promptName = collectionName dans Chat.tsx**

Dans `Chat.tsx` ligne 172 :
```typescript
for await (const event of streamChat(content, collection, collection, history)) {
```
Le 3e argument (`promptName`) vaut `collection` (nom de la collection), pas `"defaut"`. Ainsi si la collection s'appelle `"vlm_robotics"`, le prompt VLM Robotics est bien sélectionné — mais si elle s'appelle `"test"` ou `"documents"`, `get_prompt("test")` retourne le prompt par défaut (fallback silencieux).

Questions :
- Est-ce un bug intentionnel (convention collection-name = prompt-name) ou une erreur de copier-coller ?
- Devrait-il y avoir un sélecteur de prompt indépendant dans l'UI ?

---

## FORMAT DE RÉPONSE ATTENDU

Pour chaque problème identifié, utilise le format suivant :

**[SEVERITE : CRITIQUE / MAJEUR / MINEUR]** Titre court du problème
- **Localisation :** `fichier.py:ligne` ou `composant:ligne`
- **Problème :** Explication claire du risque concret (comportement observable, pas seulement théorique)
- **Solution :** Code concret ou pattern recommandé (snippet Python/TypeScript si pertinent)

Termine ta réponse par un **tableau de priorisation** listant les **5 à 8 améliorations** avec le meilleur ratio impact/effort, sous cette forme :

| Priorité | Amélioration | Sévérité | Effort estimé | Impact |
|----------|-------------|---------|--------------|--------|
| 1 | ... | CRITIQUE | 1h | Évite X |
| 2 | ... | MAJEUR | 2h | Améliore Y |
| ... | | | | |

---

*Prompt généré le 2026-02-19 pour un audit complet du projet chatbot-local.*
*Tous les chemins de fichiers sont relatifs à `/home/lucmc94/chatbot-local/`.*
