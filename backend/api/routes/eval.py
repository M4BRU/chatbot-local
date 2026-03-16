"""Eval API routes — accès aux résultats RAGAS stockés en SQLite."""

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/eval", tags=["eval"])


@router.get("/stats")
async def get_eval_stats() -> dict:
    """Statistiques agrégées sur toutes les évaluations."""
    from backend.core.eval_store import get_stats, get_pending_count
    stats = get_stats()
    stats["pending"] = get_pending_count()
    return stats


@router.get("/queue-status")
async def queue_status() -> dict:
    """Nombre d'items en attente d'évaluation + état du batch en cours."""
    from backend.core.eval_store import get_pending_count
    from backend.core.evaluator import is_batch_running
    return {"pending": get_pending_count(), "running": is_batch_running()}


@router.post("/run-batch")
async def run_batch(limit: int = Query(default=5, ge=1, le=50)) -> dict:
    """Lance l'évaluation RAGAS sur jusqu'à `limit` items en attente."""
    from backend.core.eval_store import get_pending_count
    from backend.core.evaluator import start_batch_eval, is_batch_running

    if is_batch_running():
        return {"started": False, "reason": "Batch déjà en cours"}

    pending = get_pending_count()
    if pending == 0:
        return {"started": False, "reason": "Aucun item en attente"}

    started = start_batch_eval(limit=limit)
    return {"started": started, "pending_before": pending, "limit": limit}


@router.get("/recent")
async def get_recent_evals(limit: int = Query(default=50, le=200)) -> list[dict]:
    """Dernières évaluations (les plus récentes en premier)."""
    from backend.core.eval_store import get_recent
    return get_recent(limit=limit)


@router.get("/trend")
async def get_eval_trend(days: int = Query(default=30, le=365)) -> list[dict]:
    """Moyennes journalières sur les N derniers jours."""
    from backend.core.eval_store import get_trend
    return get_trend(days=days)


@router.get("/breakdown")
async def get_hash_breakdown() -> list[dict]:
    """Stats groupées par (pipeline_hash, search_hash)."""
    from backend.core.eval_store import get_hash_breakdown
    return get_hash_breakdown()
