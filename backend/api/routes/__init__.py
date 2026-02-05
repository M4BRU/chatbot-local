"""API routes."""

from .chat import router as chat_router
from .collections import router as collections_router
from .documents import router as documents_router
from .health import router as health_router

__all__ = ["health_router", "chat_router", "collections_router", "documents_router"]
