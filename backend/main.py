"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.dependencies import get_settings
from backend.api.routes import agent_router, chat_router, collections_router, conversations_router, devis_router, documents_router, excel_documents_router, eval_router, health_router

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


def _warmup_ollama() -> None:
    """
    Pre-charge les modèles Ollama (embed + LLM) pour que la première requête
    utilisateur soit rapide. Exécuté en thread de fond au démarrage.
    """
    from core.embeddings import OLLAMA_API_GENERATE, OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDING_MODEL, USE_HF_EMBEDDINGS

    # 1. Warm-up embed
    if USE_HF_EMBEDDINGS:
        # HF embeddings : charger le modèle sentence-transformers directement.
        # Ne PAS appeler l'endpoint Ollama embed — ça chargerait mxbai inutilement
        # dans Ollama (96 Mo VRAM + 601 Mo RAM) alors qu'on ne l'utilise plus depuis Ollama.
        try:
            from core.embeddings import get_embeddings
            get_embeddings().embed_query("warmup")
            logger.info("Warmup HF embeddings OK")
        except Exception as e:
            logger.warning(f"Warmup HF embeddings échoué : {e}")
    else:
        try:
            requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": "warmup"},
                timeout=60,
            )
            logger.info("Warmup embed OK")
        except Exception as e:
            logger.warning(f"Warmup embed échoué : {e}")

    # 2. Warm-up LLM : charge le modèle en mémoire sans générer.
    # timeout=600 : 10 min max. Évite un thread zombie si Ollama crash définitivement.
    # Le chargement de llama3.1:8b peut dépasser 5 min sur certaines machines,
    # un timeout court ferme la connexion et fait abandonner le chargement à Ollama
    # ("client connection closed before server finished loading").
    try:
        requests.post(
            OLLAMA_API_GENERATE,
            json={"model": OLLAMA_MODEL, "prompt": "warmup", "stream": False,
                  "options": {"num_predict": 1}},
            timeout=600,
        )
        logger.info("Warmup LLM OK")
    except Exception as e:
        logger.warning(f"Warmup LLM échoué : {e}")


def _warmup_reranker() -> None:
    """Pre-charge le reranker actif (BGE ou ColBERT) en mémoire."""
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
    # Init SQLite DBs
    from backend.api.dependencies import get_conversation_manager, get_catalog_adapter, get_excel_collection_adapter
    get_conversation_manager().init_db()
    get_catalog_adapter().init_db()  # creates devis_paniers table
    get_excel_collection_adapter().init_db()  # creates excel_collections table
    from backend.core.eval_store import init_eval_db
    init_eval_db()  # creates eval_log table

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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(collections_router)
app.include_router(conversations_router)
app.include_router(documents_router)
app.include_router(excel_documents_router)
app.include_router(devis_router)
app.include_router(eval_router)
app.include_router(agent_router)


@app.get("/")
async def root():
    """Root endpoint - basic JSON response."""
    return {"message": "chatbot-local API", "version": "0.1.0"}
