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


async def check_chromadb(settings: Settings) -> str:
    """Check ChromaDB service health."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"http://{settings.chroma_host}:{settings.chroma_port}/api/v1/heartbeat"
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


@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)) -> ApiResponse:
    """Check system health.

    Returns status of Ollama, ChromaDB, and GPU detection.
    Always returns HTTP 200 - degraded status is indicated in the response body.
    """
    ollama_status = await check_ollama(settings)
    chromadb_status = await check_chromadb(settings)
    gpu_status = check_gpu()

    data = {
        "ollama": ollama_status,
        "chromadb": chromadb_status,
        "gpu": gpu_status,
    }

    # Build degradation message
    degraded = []
    if ollama_status == "unavailable":
        degraded.append("Ollama unreachable")
    if chromadb_status == "unavailable":
        degraded.append("ChromaDB unreachable")

    message = f"Degraded: {', '.join(degraded)}" if degraded else None

    return ApiResponse.success(data=data, message=message)
