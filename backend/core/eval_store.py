"""SQLite store for RAG evaluation results (RAGAS + Level A metrics)."""

import json
import os
import sqlite3
from datetime import datetime

_DB_PATH = os.environ.get("EVAL_DB_PATH", "/app/documents/eval_log.db")


def init_eval_db() -> None:
    """Create the eval_log and eval_queue tables if they don't exist."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT NOT NULL,
                collection        TEXT,
                pipeline_hash     TEXT,
                search_hash       TEXT,
                question          TEXT,
                answer_preview    TEXT,
                context_preview   TEXT,
                faithfulness      REAL,
                answer_relevancy  REAL,
                context_precision REAL,
                retrieval_ms      REAL,
                eval_ms           REAL,
                eval_model        TEXT,
                eval_error        TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_queue (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                collection     TEXT NOT NULL,
                pipeline_hash  TEXT,
                search_hash    TEXT,
                question       TEXT NOT NULL,
                answer         TEXT NOT NULL,
                context_chunks TEXT NOT NULL,
                retrieval_ms   REAL
            )
        """)
        conn.commit()


def insert_eval(
    *,
    collection: str,
    pipeline_hash: str,
    search_hash: str,
    question: str,
    answer_preview: str,
    context_preview: str,
    faithfulness: float | None,
    answer_relevancy: float | None,
    context_precision: float | None,
    retrieval_ms: float,
    eval_ms: float,
    eval_model: str,
    eval_error: str | None,
) -> None:
    """Insert one evaluation row."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """INSERT INTO eval_log
               (timestamp, collection, pipeline_hash, search_hash,
                question, answer_preview, context_preview,
                faithfulness, answer_relevancy, context_precision,
                retrieval_ms, eval_ms, eval_model, eval_error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                collection,
                pipeline_hash,
                search_hash,
                question[:500],
                answer_preview[:500],
                context_preview[:500],
                faithfulness,
                answer_relevancy,
                context_precision,
                retrieval_ms,
                eval_ms,
                eval_model,
                eval_error,
            ),
        )
        conn.commit()


def get_stats() -> dict:
    """Aggregate stats over all eval_log rows."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                AVG(faithfulness)       AS avg_faithfulness,
                AVG(answer_relevancy)   AS avg_answer_relevancy,
                AVG(context_precision)  AS avg_context_precision,
                AVG(retrieval_ms)       AS avg_retrieval_ms,
                AVG(eval_ms)            AS avg_eval_ms,
                SUM(CASE WHEN eval_error IS NOT NULL THEN 1 ELSE 0 END) AS errors
            FROM eval_log
        """).fetchone()
        return dict(row) if row else {}


def get_recent(limit: int = 50) -> list[dict]:
    """Return the most recent eval rows."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM eval_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_trend(days: int = 30) -> list[dict]:
    """Daily averages for the last N days."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                date(timestamp)        AS day,
                COUNT(*)               AS count,
                AVG(faithfulness)      AS faithfulness,
                AVG(answer_relevancy)  AS answer_relevancy,
                AVG(context_precision) AS context_precision
            FROM eval_log
            WHERE timestamp >= date('now', ?)
            GROUP BY day
            ORDER BY day
        """, (f"-{days} days",)).fetchall()
        return [dict(r) for r in rows]


# ─── eval_queue functions ────────────────────────────────────────────────────


def enqueue_raw(
    *,
    collection: str,
    pipeline_hash: str,
    search_hash: str,
    question: str,
    answer: str,
    context_chunks: list[str],
    retrieval_ms: float,
) -> None:
    """Stocke les données brutes dans eval_queue pour évaluation différée."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """INSERT INTO eval_queue
               (timestamp, collection, pipeline_hash, search_hash,
                question, answer, context_chunks, retrieval_ms)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                collection,
                pipeline_hash,
                search_hash,
                question,
                answer,
                json.dumps(context_chunks),
                retrieval_ms,
            ),
        )
        conn.commit()


def get_pending_count() -> int:
    """Nombre d'évaluations en attente dans eval_queue."""
    with sqlite3.connect(_DB_PATH) as conn:
        row = conn.execute("SELECT COUNT(*) FROM eval_queue").fetchone()
        return row[0] if row else 0


def get_pending_batch(limit: int = 5) -> list[dict]:
    """Retourne jusqu'à `limit` items en attente d'évaluation."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM eval_queue ORDER BY id ASC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            item["context_chunks"] = json.loads(item["context_chunks"])
            result.append(item)
        return result


def delete_queued(ids: list[int]) -> None:
    """Supprime les items traités de eval_queue."""
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(f"DELETE FROM eval_queue WHERE id IN ({placeholders})", ids)
        conn.commit()


def get_hash_breakdown() -> list[dict]:
    """Stats grouped by (pipeline_hash, search_hash)."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                pipeline_hash,
                search_hash,
                COUNT(*)               AS count,
                AVG(faithfulness)      AS faithfulness,
                AVG(answer_relevancy)  AS answer_relevancy,
                AVG(context_precision) AS context_precision,
                MIN(timestamp)         AS first_seen,
                MAX(timestamp)         AS last_seen
            FROM eval_log
            GROUP BY pipeline_hash, search_hash
            ORDER BY last_seen DESC
        """).fetchall()
        return [dict(r) for r in rows]
