"""API routes."""

from .agent import router as agent_router
from .chat import router as chat_router
from .collections import router as collections_router
from .conversations import router as conversations_router
from .devis import router as devis_router
from .documents import router as documents_router
from .documents import excel_router as excel_documents_router
from .eval import router as eval_router
from .health import router as health_router
from .transcription import router as transcription_router

__all__ = [
    "health_router",
    "chat_router",
    "collections_router",
    "conversations_router",
    "devis_router",
    "documents_router",
    "excel_documents_router",
    "eval_router",
    "agent_router",
    "transcription_router",
]
