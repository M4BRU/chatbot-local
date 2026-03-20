"""Health check endpoint."""

import subprocess

import httpx
from fastapi import APIRouter, Depends

from backend.api.dependencies import get_settings
from backend.config import Settings
from backend.domain.models import ApiResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


async def check_ollama(settings: Settings) -> str:
    """Check Ollama service health."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            if response.status_code == 200:
                return "ok"
            return "unavailable"
    except Exception:
        return "unavailable"


async def check_qdrant(settings: Settings) -> str:
    """Check Qdrant service health."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"http://{settings.qdrant_host}:{settings.qdrant_port}/healthz"
            )
            if response.status_code == 200:
                return "ok"
            return "unavailable"
    except Exception:
        return "unavailable"


def check_gpu() -> str:
    """Check GPU availability via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "detected"
        return "not_detected"
    except Exception:
        return "not_detected"


@router.get("/status")
async def llm_status(settings: Settings = Depends(get_settings)) -> dict:
    """Check if the LLM model is available in Ollama (lazy-loaded on first request)."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_base = settings.ollama_model.split(":")[0]
                ready = any(model_base in m.get("name", "") for m in models)
                return {"llm_ready": ready}
    except Exception:
        pass
    return {"llm_ready": False}


@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)) -> ApiResponse:
    """Check system health — Ollama, Qdrant et GPU."""
    ollama_status = await check_ollama(settings)
    qdrant_status = await check_qdrant(settings)
    gpu_status = check_gpu()

    data = {
        "ollama": ollama_status,
        "qdrant": qdrant_status,
        "gpu": gpu_status,
    }

    degraded = []
    if ollama_status == "unavailable":
        degraded.append("Ollama unreachable")
    if qdrant_status == "unavailable":
        degraded.append("Qdrant unreachable")

    message = f"Degraded: {', '.join(degraded)}" if degraded else None

    return ApiResponse.success(data=data, message=message)
