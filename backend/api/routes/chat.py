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
    message: str, collection_name: str, prompt_name: str, history: list, conv_id: str | None = None
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
                answer_parts: list[str] = []
                for token in result["reponse"]:
                    answer_parts.append(token)
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))

                # Stocker dans eval_queue pour évaluation différée
                metrics = result.get("metrics", {})
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
                except Exception:
                    pass

                # Persistance backend (résiste aux déconnexions SSE)
                if conv_id:
                    try:
                        from backend.core.conversation_manager import ConversationManager
                        cm = ConversationManager()
                        cm.add_message(conv_id, "user", message)
                        if answer_parts:
                            cm.add_message(conv_id, "assistant", "".join(answer_parts))
                    except Exception:
                        pass

                loop.call_soon_threadsafe(queue.put_nowait, ("done", {
                    "sources": result["sources"],
                    "metrics": metrics,
                }))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        while True:
            try:
                kind, data = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Keepalive SSE : évite que le browser ferme la connexion pendant
                # le chargement du modèle Ollama (cold start pouvant dépasser 60s)
                yield ": keepalive\n\n"
                continue
            if kind == "token":
                yield f"data: {json.dumps({'token': data})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'sources': data['sources'], 'metrics': data.get('metrics'), 'done': True})}\n\n"
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
            request.history,
            request.conv_id,
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
