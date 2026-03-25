"""Chat API routes with SSE streaming."""

import asyncio
import json
import logging
import threading
import uuid as _uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from backend.adapters.auth_adapter import get_current_user
from backend.api.dependencies import get_authorization_service, get_collection_manager

logger = logging.getLogger(__name__)
from backend.db.models import Conversation, Message, UserTable
from backend.domain.models.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])

# Maximum number of history messages to include
MAX_HISTORY_MESSAGES = 10


def _format_history(history: list) -> str:
    """Format conversation history for the prompt."""
    if not history:
        return ""

    formatted = []
    # Take only the last N messages
    recent_history = history[-MAX_HISTORY_MESSAGES:]

    for msg in recent_history:
        role = "User" if msg.role == "user" else "Assistant"
        formatted.append(f"{role}: {msg.content}")

    return "\n".join(formatted)


async def _persist_messages(
    conv_id: str,
    user_id: _uuid.UUID,
    user_message: str,
    assistant_message: str,
) -> None:
    """Persiste les messages dans PostgreSQL (async, hors thread)."""
    try:
        conv_uuid = _uuid.UUID(conv_id)
    except ValueError:
        return
    try:
        from backend.db.base import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Conversation).where(
                    Conversation.id == conv_uuid,
                    Conversation.user_id == user_id,
                )
            )
            conv = result.scalar_one_or_none()
            if conv:
                session.add(Message(conversation_id=conv.id, role="user", content=user_message))
                if assistant_message:
                    session.add(Message(conversation_id=conv.id, role="assistant", content=assistant_message))
                conv.updated_at = datetime.now(timezone.utc)
                await session.commit()
    except Exception as e:
        logger.error("Failed to persist messages for conversation %s: %s", conv_id, e)


async def _stream_rag_response(
    message: str,
    collection_name: str,
    prompt_name: str,
    history: list,
    conv_id: str | None = None,
    reasoning_mode: bool = False,
    user_id: _uuid.UUID | None = None,
) -> AsyncGenerator[str, None]:
    """Stream RAG response as SSE events.

    Le RAG (recherche ChromaDB + génération Ollama) tourne dans un thread dédié
    via asyncio.Queue + call_soon_threadsafe pour ne pas bloquer la boucle d'événements.
    La persistance des messages se fait côté async (PostgreSQL) après réception du done.
    """
    from core.search import RAGEngine, ReasoningEngine

    cm = get_collection_manager()
    if not cm.collection_existe(collection_name):
        yield f"data: {json.dumps({'error': f'Collection {collection_name} not found'})}\n\n"
        return

    try:
        history_text = _format_history(history)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_in_thread() -> None:
            """Exécute le pipeline RAG dans un thread pour ne pas bloquer l'event loop."""
            try:
                answer_parts: list[str] = []
                sources: list = []
                metrics: dict = {}

                if reasoning_mode:
                    engine = ReasoningEngine(collection_name, prompt_name=prompt_name, collection_manager=cm)

                    def _cb(kind, data):
                        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

                    for kind, data in engine.stream_with_reasoning(message, history_text, _cb):
                        if kind == "token":
                            answer_parts.append(data)
                            loop.call_soon_threadsafe(queue.put_nowait, ("token", data))
                        elif kind == "done":
                            sources = data.get("sources", [])
                            metrics = data.get("metrics", {})
                else:
                    rag = RAGEngine(collection_name, prompt_name=prompt_name, collection_manager=cm)
                    result = rag.generer_avec_sources(message, stream=True, history=history_text)
                    metrics = result.get("metrics", {})

                    for token in result["reponse"]:
                        answer_parts.append(token)
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))

                    sources = result["sources"]

                    # Stocker dans eval_queue pour évaluation différée
                    try:
                        from backend.core.evaluator import enqueue_eval
                        enqueue_eval(
                            collection=collection_name,
                            pipeline_hash=metrics.get("pipeline_hash", "unknown"),
                            search_hash=metrics.get("search_hash", "unknown"),
                            question=message,
                            answer="".join(answer_parts),
                            context_chunks=result.get("context_chunks", []),
                            retrieval_ms=metrics.get("retrieval_ms", 0),
                        )
                    except Exception as e:
                        logger.warning("Failed to enqueue evaluation: %s", e)

                loop.call_soon_threadsafe(queue.put_nowait, ("done", {
                    "sources": sources,
                    "metrics": metrics,
                    "answer_text": "".join(answer_parts),
                }))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        while True:
            try:
                kind, data = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                # Keepalive SSE : vrai event data: (pas un commentaire) pour que
                # Uvicorn, les proxies et le browser flush réellement le stream.
                # Le frontend ignore les événements avec keepalive=True.
                yield f"data: {json.dumps({'keepalive': True})}\n\n"
                continue
            if kind == "token":
                yield f"data: {json.dumps({'token': data})}\n\n"
            elif kind == "reasoning_step":
                yield f"data: {json.dumps({'reasoning_step': data})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'sources': data['sources'], 'metrics': data.get('metrics'), 'done': True})}\n\n"
                # Persistance async PostgreSQL (remplace l'ancien ConversationManager synchrone)
                if conv_id and user_id:
                    await _persist_messages(conv_id, user_id, message, data.get("answer_text", ""))
                break
            elif kind == "error":
                yield f"data: {json.dumps({'error': data})}\n\n"
                break

    except ValueError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Internal error: {str(e)}'})}\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: UserTable = Depends(get_current_user),
) -> StreamingResponse:
    """
    Chat endpoint with RAG and SSE streaming.

    Searches the specified collection for relevant context,
    then streams the LLM response token by token.
    """
    get_authorization_service().assert_can_access(current_user.role, request.collection_name)
    return StreamingResponse(
        _stream_rag_response(
            request.message,
            request.collection_name,
            request.prompt_name,
            request.history,
            request.conv_id,
            request.reasoning_mode,
            user_id=current_user.id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/debug")
async def chat_debug(
    request: ChatRequest,
    current_user: UserTable = Depends(get_current_user),
) -> dict:
    """
    Debug endpoint : retourne les chunks récupérés + query rewriting, sans appeler le LLM.
    Utile pour diagnostiquer le pipeline RAG.
    """
    from core.search import RAGEngine

    get_authorization_service().assert_can_access(current_user.role, request.collection_name)

    cm = get_collection_manager()
    if not cm.collection_existe(request.collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found")

    try:
        rag = RAGEngine(request.collection_name, prompt_name=request.prompt_name, collection_manager=cm)
        history_text = _format_history(request.history)
        result = await asyncio.to_thread(rag.rechercher_debug, request.message, history_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(
    request: ChatRequest,
    current_user: UserTable = Depends(get_current_user),
) -> ChatResponse:
    """
    Non-streaming chat endpoint for testing.

    Returns the complete response at once.
    """
    from core.search import RAGEngine

    get_authorization_service().assert_can_access(current_user.role, request.collection_name)

    cm = get_collection_manager()
    if not cm.collection_existe(request.collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found")

    try:
        rag = RAGEngine(request.collection_name, prompt_name=request.prompt_name, collection_manager=cm)
        history_text = _format_history(request.history)
        result = rag.generer_avec_sources(request.message, stream=False, history=history_text)
        return ChatResponse(response=result["reponse"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
