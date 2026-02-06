"""Chat API routes with SSE streaming."""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

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
    """Stream RAG response as SSE events."""
    from core.collection_manager import CollectionManager
    from core.search import RAGEngine

    try:
        cm = CollectionManager()
        if not cm.collection_existe(collection_name):
            yield f"data: {json.dumps({'error': f'Collection {collection_name} not found'})}\n\n"
            return

        rag = RAGEngine(collection_name, prompt_name=prompt_name, collection_manager=cm)

        # Format history for context
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
