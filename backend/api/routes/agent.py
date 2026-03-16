"""Agent orchestrator route — mode Agent.

POST /api/v1/agent/chat
SSE stream : classification → routing → réponse (Claude | RAG | combined).
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.api.dependencies import get_settings

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class AgentChatRequest(BaseModel):
    message: str
    collection_name: str = "default"
    force_mode: Literal["simple_claude", "simple_gpt", "simple_rag", "combined", "combined_gpt"] | None = None


async def _stream_with_keepalive(gen: AsyncGenerator[str, None], timeout: float = 15.0):
    """Wraps an async SSE generator with keepalive comments (évite le browser timeout)."""
    aiter = gen.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(aiter.__anext__(), timeout=timeout)
            yield chunk
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
        except StopAsyncIteration:
            break


def _get_orchestrator():
    settings = get_settings()
    from backend.domain.services.orchestrator_service import OrchestratorService
    return OrchestratorService(
        api_key=settings.anthropic_api_key,
        claude_model=settings.claude_model,
        max_tokens=settings.agent_max_tokens,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )


@router.post("/chat")
async def agent_chat(request: AgentChatRequest) -> StreamingResponse:
    """
    Agent orchestrator endpoint.

    Classifie l'intention, route vers Claude API, RAG local, ou les deux.
    Aucune donnée VLM n'est transmise à l'API externe.
    """
    orchestrator = _get_orchestrator()

    return StreamingResponse(
        _stream_with_keepalive(orchestrator.stream_response(
            message=request.message,
            collection_name=request.collection_name,
            force_mode=request.force_mode,
        )),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
