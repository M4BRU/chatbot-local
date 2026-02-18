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
    """
    Pre-charge les modèles Ollama (embed + LLM) pour que la première requête
    utilisateur soit rapide. Exécuté en thread de fond au démarrage.
    """
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

    # 2. Warm-up LLM (num_predict=1 pour charger le modèle sans générer)
    try:
        requests.post(
            OLLAMA_API_GENERATE,
            json={"model": "llama3.1:8b", "prompt": "warmup", "stream": False,
                  "options": {"num_predict": 1}},
            timeout=120,
        )
        logger.info("Warmup LLM OK")
    except Exception as e:
        logger.warning(f"Warmup LLM échoué : {e}")


def _warmup_reranker() -> None:
    """Pre-charge le reranker BGE (FlagEmbedding) en mémoire."""
    try:
        from core.search import _get_reranker
        _get_reranker()
        logger.info("Warmup reranker OK")
    except Exception as e:
        logger.warning(f"Warmup reranker échoué : {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lance le pre-warm Ollama + reranker en arrière-plan au démarrage."""
    loop = asyncio.get_event_loop()
    logger.info("Démarrage pre-warm modèles (arrière-plan)…")
    loop.run_in_executor(None, _warmup_ollama)
    loop.run_in_executor(None, _warmup_reranker)
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
app.include_router(documents_router)


@app.get("/")
async def root():
    """Root endpoint - basic JSON response."""
    return {"message": "chatbot-local API", "version": "0.1.0"}
