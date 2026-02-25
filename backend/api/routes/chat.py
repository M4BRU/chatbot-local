"""Chat API routes with SSE streaming."""

import asyncio
import json
import threading
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_collection_manager
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


async def _stream_rag_response(
    message: str, collection_name: str, prompt_name: str, history: list
) -> AsyncGenerator[str, None]:
    """Stream RAG response as SSE events.

    Le RAG (recherche ChromaDB + génération Ollama) tourne dans un thread dédié
    via asyncio.Queue + call_soon_threadsafe pour ne pas bloquer la boucle d'événements.
    """
    from core.search import RAGEngine

    cm = get_collection_manager()
    if not cm.collection_existe(collection_name):
        yield f"data: {json.dumps({'error': f'Collection {collection_name} not found'})}\n\n"
        return

    try:
        rag = RAGEngine(collection_name, prompt_name=prompt_name, collection_manager=cm)
        history_text = _format_history(history)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_in_thread() -> None:
            """Exécute le pipeline RAG dans un thread pour ne pas bloquer l'event loop."""
            try:
                result = rag.generer_avec_sources(message, stream=True, history=history_text)
                for token in result["reponse"]:
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", result["sources"]))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        while True:
            kind, data = await queue.get()
            if kind == "token":
                yield f"data: {json.dumps({'token': data})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'sources': data, 'done': True})}\n\n"
                break
            elif kind == "error":
                yield f"data: {json.dumps({'error': data})}\n\n"
                break

    except ValueError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Internal error: {str(e)}'})}\n\n"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Chat endpoint with RAG and SSE streaming.

    Searches the specified collection for relevant context,
    then streams the LLM response token by token.
    """
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


@router.post("/chat/debug")
async def chat_debug(request: ChatRequest) -> dict:
    """
    Debug endpoint : retourne les chunks récupérés + query rewriting, sans appeler le LLM.
    Utile pour diagnostiquer le pipeline RAG.
    """
    from core.search import RAGEngine

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
async def chat_sync(request: ChatRequest) -> ChatResponse:
    """
    Non-streaming chat endpoint for testing.

    Returns the complete response at once.
    """
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
