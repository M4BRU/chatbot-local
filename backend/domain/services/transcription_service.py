"""Service de transcription audio — appel Whisper + résumé structuré LLM."""

import json
import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:9000")
WHISPER_TIMEOUT = int(os.getenv("WHISPER_TIMEOUT", "600"))

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}
MAX_FILE_SIZE_MB = 300

# ── Shared httpx.AsyncClient ────────────────────────────────────────────────
_HTTP_CLIENT: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Retourne un AsyncClient partagé (connection pooling, pas de leak TCP)."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=WHISPER_TIMEOUT)
    return _HTTP_CLIENT


# ── JSON schema pour le résumé structuré (Ollama format-constrained) ────────
_SCHEMA_SUMMARY = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "responsable": {"type": "string"},
                    "description": {"type": "string"},
                    "deadline": {"type": "string"},
                },
                "required": ["description"],
            },
        },
        "participants": {
            "type": "array",
            "items": {"type": "string"},
        },
        "points_cles": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["decisions", "actions", "participants", "points_cles"],
}

_SUMMARY_SYSTEM_PROMPT = """\
Tu es un assistant spécialisé dans la rédaction de comptes-rendus de réunion.

À partir de la transcription ci-dessous, extrais un résumé structuré en JSON avec les clés suivantes :
- **decisions** : liste des décisions prises pendant la réunion
- **actions** : liste des actions à faire, chacune avec "description" (obligatoire), "responsable" (si mentionné), "deadline" (si mentionné)
- **participants** : liste des noms ou rôles des participants détectés
- **points_cles** : liste des points clés discutés

Règles :
- Réponds UNIQUEMENT en JSON valide, sans texte avant ou après.
- Si une information n'est pas mentionnée, utilise une liste vide [].
- Conserve la langue de la transcription (français par défaut).
- Ne fabrique aucune information non présente dans la transcription.
"""


class TranscriptionService:
    """Orchestre la transcription audio (Whisper) et le résumé structuré (LLM)."""

    async def transcribe_audio(self, file_path: Path) -> dict:
        """Envoie le fichier audio au service Whisper et retourne la transcription."""
        client = _get_http_client()
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            resp = await client.post(
                f"{WHISPER_URL}/transcribe",
                files=files,
                timeout=WHISPER_TIMEOUT,
            )
        resp.raise_for_status()
        return resp.json()

    async def summarize_transcript(self, transcript_text: str) -> dict:
        """Génère un résumé structuré via Ollama (format JSON contraint)."""
        if not transcript_text.strip():
            return {"decisions": [], "actions": [], "participants": [], "points_cles": []}

        # Tronque si trop long pour le contexte LLM (garde ~6000 tokens ≈ 24000 chars)
        max_chars = 24000
        text = transcript_text[:max_chars]
        if len(transcript_text) > max_chars:
            text += "\n\n[... transcription tronquée pour le résumé]"

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcription :\n\n{text}"},
            ],
            "stream": False,
            "think": False,
            "format": _SCHEMA_SUMMARY,
            "options": {"temperature": 0, "num_ctx": 8192},
        }

        client = _get_http_client()
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "").strip()
        logger.info("[transcription] Résumé LLM brut (%d chars)", len(content))

        return json.loads(content)

    async def process_audio_stream(
        self, file_path: Path, filename: str
    ) -> AsyncGenerator[dict, None]:
        """Async generator SSE : transcription Whisper → résumé LLM."""
        try:
            # Phase 1 : Transcription
            yield {"status": "transcribing", "step": "Transcription en cours…"}

            result = await self.transcribe_audio(file_path)
            transcript = result.get("text", "")
            segments = result.get("segments", [])
            duration = result.get("duration_seconds", 0)

            logger.info(
                "[transcription] Whisper done: %d segments, %.1fs",
                len(segments), duration,
            )

            yield {
                "status": "transcribed",
                "transcript": transcript,
                "segments": segments,
                "duration_seconds": duration,
            }

            # Phase 2 : Résumé structuré
            if not transcript.strip():
                yield {
                    "status": "done",
                    "summary": {
                        "decisions": [],
                        "actions": [],
                        "participants": [],
                        "points_cles": ["Aucune parole détectée dans l'audio."],
                    },
                }
                return

            yield {"status": "summarizing", "step": "Génération du résumé…"}

            summary = await self.summarize_transcript(transcript)
            logger.info("[transcription] Summary done: %s", list(summary.keys()))

            yield {"status": "done", "summary": summary}

        except httpx.HTTPStatusError as e:
            logger.error("[transcription] HTTP error: %s", e)
            yield {"error": f"Erreur du service de transcription : {e.response.status_code}"}
        except httpx.ConnectError:
            logger.error("[transcription] Whisper service unreachable at %s", WHISPER_URL)
            yield {"error": "Service de transcription indisponible. Vérifiez que le container whisper est démarré."}
        except Exception as e:
            logger.error("[transcription] Unexpected error: %s", e, exc_info=True)
            yield {"error": f"Erreur inattendue : {str(e)}"}
        finally:
            if file_path.exists():
                file_path.unlink(missing_ok=True)
