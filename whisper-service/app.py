"""Whisper transcription micro-service — CPU-only, anti-hallucination."""

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("whisper-service")

# ── Configuration via env vars (swappable sans rebuild) ──────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "fr")
BLOCKLIST_PATH = Path(__file__).parent / "blocklist_fr.txt"

# ── Globals (loaded on startup) ──────────────────────────────────────────────
_model = None
_blocklist: set[str] = set()

app = FastAPI(title="whisper-service", version="0.1.0")


def _load_blocklist() -> set[str]:
    """Charge les phrases hallucinees connues (une par ligne)."""
    if not BLOCKLIST_PATH.exists():
        return set()
    lines = BLOCKLIST_PATH.read_text(encoding="utf-8").splitlines()
    return {line.strip().lower() for line in lines if line.strip()}


@app.on_event("startup")
def startup():
    """Charge le modele Whisper et la blocklist au demarrage."""
    global _model, _blocklist
    from faster_whisper import WhisperModel

    logger.info(
        "Loading model=%s device=%s compute=%s",
        WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    )
    _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    _blocklist = _load_blocklist()
    logger.info("Model loaded. Blocklist: %d phrases", len(_blocklist))


def _is_hallucination(text: str) -> bool:
    """Verifie si un segment est une hallucination connue."""
    cleaned = text.strip().lower().rstrip(".")
    return cleaned in _blocklist


def _filter_repeated_segments(segments: list[dict], max_repeats: int = 3) -> list[dict]:
    """Supprime les segments repetes consecutivement (boucles infinies Whisper)."""
    if not segments:
        return segments
    filtered = [segments[0]]
    repeat_count = 1
    for seg in segments[1:]:
        if seg["text"].strip() == filtered[-1]["text"].strip():
            repeat_count += 1
            if repeat_count <= max_repeats:
                filtered.append(seg)
        else:
            repeat_count = 1
            filtered.append(seg)
    return filtered


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcrit un fichier audio avec anti-hallucination."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    suffix = Path(file.filename or "audio").suffix.lower() or ".wav"
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        tmp_path = Path(tmp.name)

        logger.info("Transcribing %s (%d bytes, lang=%s)", file.filename, len(content), WHISPER_LANGUAGE)

        segments_gen, info = _model.transcribe(
            str(tmp_path),
            language=WHISPER_LANGUAGE,
            condition_on_previous_text=False,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"threshold": 0.5, "min_silence_duration_ms": 500},
        )

        # Materialise les segments et filtre les hallucinations
        raw_segments = []
        for seg in segments_gen:
            if _is_hallucination(seg.text):
                logger.debug("Blocklist hit: '%s'", seg.text.strip())
                continue
            raw_segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text,
            })

        # Filtre les repetitions (boucles infinies)
        segments = _filter_repeated_segments(raw_segments)

        full_text = " ".join(s["text"].strip() for s in segments)
        duration = info.duration if info.duration else 0.0

        logger.info(
            "Done: %d segments, %.1fs, filtered %d hallucinations",
            len(segments), duration, len(raw_segments) - len(segments),
        )

        return {
            "text": full_text,
            "segments": segments,
            "language": info.language,
            "duration_seconds": round(duration, 1),
        }

    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "ok" if _model is not None else "loading",
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
    }
