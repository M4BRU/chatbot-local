"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from backend.api.dependencies import get_settings
from backend.api.routes import (
    agent_router,
    chat_router,
    collections_router,
    conversations_router,
    devis_router,
    documents_router,
    excel_documents_router,
    eval_router,
    health_router,
)
from backend.api.routes.auth import router as auth_router
from backend.api.routes.totp import router as totp_router

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

# ── Rate limiter (slowapi) ────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


def _warmup_ollama() -> None:
    """
    Pre-charge les modèles Ollama (embed + LLM) pour que la première requête
    utilisateur soit rapide. Exécuté en thread de fond au démarrage.
    """
    from core.embeddings import OLLAMA_API_GENERATE, OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDING_MODEL, USE_HF_EMBEDDINGS

    # 1. Warm-up embed
    if USE_HF_EMBEDDINGS:
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


async def _run_migrations() -> None:
    """Lance les migrations Alembic de façon synchrone bloquante (avant yield)."""
    from alembic import command as alembic_command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    await asyncio.to_thread(alembic_command.upgrade, alembic_cfg, "head")
    logger.info("Migrations Alembic appliquées")


def _check_rbac_config() -> None:
    """Vérifie que le fichier rbac.yaml est présent et valide au démarrage."""
    from backend.domain.services.authorization_service import AuthorizationService
    AuthorizationService(settings.rbac_config_path)  # lève RuntimeError si invalide
    logger.info("RBAC config OK")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise la DB (Alembic), vérifie le RBAC, puis lance les warmups."""
    # 1. Migrations PostgreSQL (bloquant — le schéma doit exister avant de servir)
    await _run_migrations()

    # 2. Validation RBAC (fail-fast si config absente ou malformée)
    await asyncio.to_thread(_check_rbac_config)

    # 3. SQLite DBs existantes (catalog, excel, eval) — inchangées
    from backend.api.dependencies import get_catalog_adapter, get_excel_collection_adapter
    get_catalog_adapter().init_db()
    get_excel_collection_adapter().init_db()
    from backend.core.eval_store import init_eval_db
    init_eval_db()

    # DEPRECATED: get_conversation_manager().init_db() — replaced by SQLAlchemy/Alembic

    # 4. Warmup modèles (arrière-plan)
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

# ── Middlewares ───────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(totp_router)
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
