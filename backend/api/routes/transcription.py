"""Transcription audio API routes — upload + SSE streaming pipeline."""

import json
import logging
import shutil
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse

from backend.api.dependencies import get_current_user, get_transcription_service
from backend.domain.services.transcription_service import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/transcription", tags=["transcription"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/upload")
async def upload_and_transcribe(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    transcription_service=Depends(get_transcription_service),
):
    """Upload un fichier audio et retourne un flux SSE (transcription + résumé)."""

    # Validation extension
    raw_name = file.filename or "audio"
    safe_name = PurePosixPath(raw_name).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés : {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Sauvegarde temporaire + validation taille
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        tmp_path = Path(tmp.name)

        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux ({size_mb:.0f} MB). Maximum : {MAX_FILE_SIZE_MB} MB.",
            )

        logger.info(
            "[transcription] Upload: %s (%.1f MB) by user %s",
            safe_name, size_mb, getattr(user, "email", "?"),
        )

    except HTTPException:
        raise
    except Exception as e:
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    # SSE stream — le service gère le cleanup du fichier temp dans son finally
    async def _generator():
        async for event in transcription_service.process_audio_stream(tmp_path, safe_name):
            yield {"data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(_generator(), headers=_SSE_HEADERS, ping=15)


@router.get("/health")
async def transcription_health(user=Depends(get_current_user)):
    """Vérifie si le service Whisper est accessible."""
    import httpx

    from backend.domain.services.transcription_service import WHISPER_URL, _get_http_client

    try:
        client = _get_http_client()
        resp = await client.get(f"{WHISPER_URL}/health", timeout=5.0)
        return resp.json()
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}
