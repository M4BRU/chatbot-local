"""
Catalog adapter: loads Excel catalog into SQLite with FTS5 full-text search
and an in-memory BM25 index (same stemmer as the RAG chat pipeline).

Price retrieval strategy
────────────────────────
CATALOG_PRICE_STRATEGY = "most_recent"
  → Uses the highest Num_Affaire as proxy for the most recent project.
  → Hypothesis: Num_Affaire is a sequential integer (higher = newer).
  → To change behavior: update CATALOG_PRICE_STRATEGY and the SQL in get_price().
"""

import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

CATALOG_DB_PATH = Path("/app/documents/catalogue.db")

# ── Price strategy ─────────────────────────────────────────────────────────────
# "most_recent" : highest Num_Affaire (assumed most recent project)
# "first"       : first matching row
CATALOG_PRICE_STRATEGY = "most_recent"

# Columns used for full-text search (normalized names, subset of Excel columns)
_FTS_COLS = ["nom_poste", "ensemble", "elements", "nom_affaire", "fournisseur"]

# ── Snowball stemmer (shared with RAG chat pipeline) ───────────────────────────
_stemmer: Any = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem.snowball import FrenchStemmer
            _stemmer = FrenchStemmer()
        except Exception as exc:
            logger.warning("Stemmer NLTK indisponible (%s) — fallback tokenisation brute", exc)
    return _stemmer


def _stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = _get_stemmer()
    if stemmer is None:
        return tokens
    return [stemmer.stem(t) for t in tokens]


def _tokenize(text: str) -> list[str]:
    return _stem_tokens(re.findall(r"\w+", text.lower()))


def _norm(name: str) -> str:
    """Normalize column name → lowercase with underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


# ── Per-column BM25 indexes ────────────────────────────────────────────────────

class _CatalogBM25Indexes:
    """One BM25 index per searchable column.

    Searching a specific column (e.g. nom_poste) only scores against that
    column's text — no cross-column noise (e.g. 'armoire' in elements no
    longer pollutes a nom_poste search).
    """

    def __init__(self, rows: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self._rows = rows
        self._indexes: dict[str, Any] = {}
        for col in _FTS_COLS:
            texts = [str(r.get(col) or "") for r in rows]
            tokenized = [_tokenize(t) for t in texts]
            self._indexes[col] = BM25Okapi(tokenized)

    def search(self, query: str, column: str = "nom_poste", limit: int = 10) -> list[dict]:
        bm25 = self._indexes.get(column) or self._indexes.get("nom_poste")
        tokens = _tokenize(query)
        scores = bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked[:limit]:
            if scores[idx] > 0:
                results.append(self._rows[idx])
        return results

    def get_all_scores(self, query: str, column: str = "nom_poste") -> list[float]:
        """Return BM25 score for every row (one float per row, same order as self._rows)."""
        bm25 = self._indexes.get(column) or self._indexes.get("nom_poste")
        tokens = _tokenize(query)
        return list(bm25.get_scores(tokens))


class CatalogAdapter:
    def __init__(self, db_path: Path = CATALOG_DB_PATH) -> None:
        self.db_path = db_path
        self._bm25_indexes: _CatalogBM25Indexes | None = None

    # ── Startup ────────────────────────────────────────────────────────────────

    def init_db(self) -> None:
        """Create tables on startup (idempotent)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devis_paniers (
                    id           TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    nom_poste    TEXT NOT NULL,
                    num_poste    TEXT,
                    ensemble     TEXT,
                    quantite     INTEGER DEFAULT 1,
                    fournisseur  TEXT,
                    fourniture   TEXT,
                    num_affaire  TEXT,
                    nom_affaire  TEXT,
                    created_at   TEXT NOT NULL
                )
            """)
            # Affaire lock: persists the chosen affaire before the first panier item
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devis_affaire_locks (
                    conversation_id TEXT PRIMARY KEY,
                    nom_affaire     TEXT NOT NULL,
                    created_at      TEXT NOT NULL
                )
            """)
            # Devis settings: global coefficient + coef_final per conversation
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devis_settings (
                    conversation_id TEXT PRIMARY KEY,
                    coefficient      REAL DEFAULT 0.0,
                    coef_final       REAL DEFAULT 0.0
                )
            """)
            # Task list: ordered list of postes the LLM planned to add
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devis_task_lists (
                    conversation_id TEXT PRIMARY KEY,
                    tasks           TEXT NOT NULL DEFAULT '[]'
                )
            """)
            # Migrations: add columns if upgrading from older schema
            for _migration in [
                "ALTER TABLE devis_paniers ADD COLUMN num_poste TEXT",
                "ALTER TABLE devis_paniers ADD COLUMN item_type TEXT NOT NULL DEFAULT 'poste'",
                "ALTER TABLE devis_affaire_locks ADD COLUMN search_all INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE devis_paniers ADD COLUMN nbre_jours_etude   INTEGER DEFAULT 0",
                "ALTER TABLE devis_paniers ADD COLUMN nbre_jours_atelier INTEGER DEFAULT 0",
                "ALTER TABLE devis_paniers ADD COLUMN nbre_jours_client  INTEGER DEFAULT 0",
                "ALTER TABLE devis_paniers ADD COLUMN is_option          INTEGER DEFAULT 0",
            ]:
                try:
                    conn.execute(_migration)
                except Exception:
                    pass  # column already exists
            conn.commit()

    def set_affaire_lock(self, conversation_id: str, nom_affaire: str) -> None:
        """Lock an affaire for this conversation (called when user picks a choice card)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO devis_affaire_locks (conversation_id, nom_affaire, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET nom_affaire = excluded.nom_affaire
                """,
                [conversation_id, nom_affaire, datetime.now(timezone.utc).isoformat()],
            )
            conn.commit()

    def set_search_scope(self, conversation_id: str, search_all: bool) -> None:
        """Set the search scope for a conversation.

        search_all=True  → next search_catalog calls return the full catalog (all affaires).
        search_all=False → search_catalog is filtered to the locked affaire (default).
        An affaire lock row must already exist; this is a no-op if none exists yet.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE devis_affaire_locks SET search_all = ? WHERE conversation_id = ?",
                [1 if search_all else 0, conversation_id],
            )
            conn.commit()

    def get_search_scope(self, conversation_id: str) -> bool:
        """Return True if the user chose 'browse all affaires' for the next search."""
        if not self.db_path.exists():
            return False
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT search_all FROM devis_affaire_locks WHERE conversation_id = ?",
                [conversation_id],
            ).fetchone()
            return bool(row and row[0])

    # ── Task list (multi-poste planning) ──────────────────────────────────────

    def set_task_list(self, conversation_id: str, items: list[str]) -> None:
        """Store the LLM's planned task list (list of poste queries to add)."""
        import json
        tasks = [{"query": q, "done": False} for q in items]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO devis_task_lists (conversation_id, tasks)
                VALUES (?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET tasks = excluded.tasks
                """,
                [conversation_id, json.dumps(tasks, ensure_ascii=False)],
            )
            conn.commit()

    def get_task_list(self, conversation_id: str) -> list[dict]:
        """Return the task list [{query, done}, ...] or [] if none."""
        import json
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT tasks FROM devis_task_lists WHERE conversation_id = ?",
                [conversation_id],
            ).fetchone()
            if not row:
                return []
            try:
                return json.loads(row[0]) or []
            except Exception:
                return []

    def complete_task_item(self, conversation_id: str, nom_poste: str, fallback_first: bool = False) -> None:
        """Mark the first pending task whose query matches nom_poste as done.

        Si fallback_first=True et qu'aucun mot ne correspond, marque la première tâche
        pending comme done. Utile quand l'utilisateur choisit un résultat catalogue via
        l'UI (le nom_poste peut différer de la query, ex: "Robot N220" pour "Comau NJ40").
        """
        import json, re as _re
        tasks = self.get_task_list(conversation_id)
        if not tasks:
            return
        nom_words = set(_re.findall(r"\w{3,}", nom_poste.lower()))
        matched = False
        for task in tasks:
            if task.get("done"):
                continue
            q_words = set(_re.findall(r"\w{3,}", task["query"].lower()))
            if nom_words & q_words:
                task["done"] = True
                matched = True
                break
        if not matched and fallback_first:
            for task in tasks:
                if not task.get("done"):
                    task["done"] = True
                    break
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE devis_task_lists SET tasks = ? WHERE conversation_id = ?",
                [json.dumps(tasks, ensure_ascii=False), conversation_id],
            )
            conn.commit()

    def clear_task_list(self, conversation_id: str) -> None:
        """Remove the task list for a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM devis_task_lists WHERE conversation_id = ?",
                [conversation_id],
            )
            conn.commit()

    # ── Excel loading ──────────────────────────────────────────────────────────

    def load_from_excel(self, excel_path: Path) -> dict:
        """
        Load Excel → SQLite main table + FTS5 virtual table + in-memory BM25 index.
        Returns metadata dict: {rows, columns, normalized_columns}.
        """
        df = pd.read_excel(excel_path)
        original_cols = df.columns.tolist()
        df.columns = [_norm(c) for c in df.columns]

        with sqlite3.connect(self.db_path) as conn:
            # Drop and recreate catalogue tables
            conn.execute("DROP TABLE IF EXISTS catalogue")
            conn.execute("DROP TABLE IF EXISTS catalogue_fts")

            # Persist raw data (SQLite rowid is implicit integer PK)
            df.to_sql("catalogue", conn, if_exists="replace", index=False)

            # Build FTS5 on available searchable columns
            available = [c for c in _FTS_COLS if c in df.columns]
            if available:
                cols_def = ", ".join(available)
                conn.execute(
                    f"CREATE VIRTUAL TABLE catalogue_fts USING fts5("
                    f"  row_id UNINDEXED, {cols_def}"
                    f")"
                )
                coalesced = ", ".join(f'COALESCE("{c}", "")' for c in available)
                conn.execute(
                    f"INSERT INTO catalogue_fts(row_id, {', '.join(available)}) "
                    f"SELECT rowid, {coalesced} FROM catalogue"
                )
            conn.commit()

        # Build per-column BM25 indexes (one index per searchable column)
        rows = df.to_dict(orient="records")
        try:
            self._bm25_indexes = _CatalogBM25Indexes(rows)
            logger.info("BM25 catalogue : index par colonne construit (%d lignes)", len(rows))
        except Exception as exc:
            self._bm25_indexes = None
            logger.warning("BM25 catalogue indisponible (%s) — fallback FTS5/LIKE", exc)

        # Count unique nom_poste values (col G from row 2)
        unique_postes = int(df["nom_poste"].dropna().nunique()) if "nom_poste" in df.columns else 0

        logger.info(
            "Catalogue chargé : %d lignes, %d colonnes, %d postes distincts",
            len(df), len(original_cols), unique_postes,
        )
        return {
            "rows": len(df),
            "unique_postes": unique_postes,
            "columns": original_cols,
            "normalized_columns": list(df.columns),
        }

    def _rebuild_bm25_if_needed(self) -> None:
        """Rebuild per-column BM25 indexes from SQLite if not in memory (e.g. after server restart)."""
        if self._bm25_indexes is not None or not self.db_path.exists():
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = [dict(r) for r in conn.execute("SELECT * FROM catalogue").fetchall()]
            if rows:
                self._bm25_indexes = _CatalogBM25Indexes(rows)
                logger.info("BM25 catalogue : index reconstruit depuis SQLite (%d lignes)", len(rows))
        except Exception as exc:
            logger.warning("BM25 catalogue : reconstruction échouée (%s)", exc)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10, column: str = "nom_poste") -> list[dict]:
        """
        Search catalog rows using per-column BM25 (Snowball-stemmed, handles plurals).
        Searches only in the specified column — no cross-column noise.
        Falls back to SQLite FTS5, then LIKE if BM25 indexes are unavailable.
        """
        if not self.db_path.exists():
            return []

        # ── Priority 1: per-column BM25 ────────────────────────────────────────
        self._rebuild_bm25_if_needed()
        if self._bm25_indexes is not None:
            results = self._bm25_indexes.search(query, column=column, limit=limit)
            if results:
                logger.info("search BM25[%s] → %d résultats pour %r", column, len(results), query)
                return results
            logger.debug("search BM25[%s] score=0 pour %r — fallback FTS5", column, query)

        # ── Priority 2: FTS5 (exact token match, no stemming) ─────────────────
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            fts_ok = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='catalogue_fts'"
            ).fetchone()

            if fts_ok and column in _FTS_COLS:
                try:
                    rows = conn.execute(
                        f"""
                        SELECT c.*
                        FROM   catalogue c
                        JOIN   catalogue_fts fts ON c.rowid = fts.row_id
                        WHERE  fts.{column} MATCH ?
                        ORDER  BY rank
                        LIMIT  ?
                        """,
                        [query, limit],
                    ).fetchall()
                    if rows:
                        return [dict(r) for r in rows]
                except sqlite3.OperationalError:
                    pass

            # ── Priority 3: LIKE fallback on the target column only ────────────
            col_exists = any(
                r[1] == column
                for r in conn.execute("PRAGMA table_info(catalogue)").fetchall()
            )
            if not col_exists:
                return []
            # Use Snowball stem for LIKE to handle plurals (armoir% → armoire/armoires)
            stem = _tokenize(query)
            pattern = f"%{stem[0]}%" if stem else f"%{query}%"
            rows = conn.execute(
                f'SELECT * FROM catalogue WHERE "{column}" LIKE ? LIMIT ?',
                [pattern, limit],
            ).fetchall()
            return [dict(r) for r in rows]

    def _bm25_get_all_scores(self, query: str, column: str = "nom_poste") -> list[float]:
        """Return BM25 score for every catalogue row. Returns [] if index unavailable."""
        self._rebuild_bm25_if_needed()
        if self._bm25_indexes is None:
            return []
        return self._bm25_indexes.get_all_scores(query, column)

    def search_hybrid(
        self, query: str, limit: int = 15, column: str = "nom_poste"
    ) -> list[dict]:
        """
        Hybrid search: merges BM25 (stemmed) + FTS5 results.

        Each row gets:
          _score      : BM25 score (float, 0.0 for FTS5-only matches)
          _confidence : 'high' (≥60% of max), 'medium' (≥25%), 'low' (>0), 'fts_only' (=0)

        Rows are deduped by (nom_poste, nom_affaire) with max score kept,
        then sorted by score DESC and truncated to `limit`.
        """
        if not self.db_path.exists():
            return []

        # ── BM25 scores for all rows ─────────────────────────────────────────
        all_scores = self._bm25_get_all_scores(query, column)
        if all_scores and self._bm25_indexes is not None:
            rows_with_score = [
                (self._bm25_indexes._rows[i], float(all_scores[i]))
                for i in range(len(all_scores))
                if all_scores[i] > 0
            ]
        else:
            rows_with_score = []

        # ── FTS5 matches ─────────────────────────────────────────────────────
        fts_rows: list[dict] = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            fts_ok = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='catalogue_fts'"
            ).fetchone()
            if fts_ok and column in _FTS_COLS:
                try:
                    rows = conn.execute(
                        f"""
                        SELECT c.*
                        FROM   catalogue c
                        JOIN   catalogue_fts fts ON c.rowid = fts.row_id
                        WHERE  fts.{column} MATCH ?
                        ORDER  BY rank
                        LIMIT  ?
                        """,
                        [query, limit * 2],
                    ).fetchall()
                    fts_rows = [dict(r) for r in rows]
                except sqlite3.OperationalError:
                    pass

        # ── Merge by (nom_poste, nom_affaire) with max score ─────────────────
        merged: dict[tuple, dict] = {}

        def _key(row: dict) -> tuple:
            return (
                (row.get("nom_poste") or "").strip().lower(),
                (row.get("nom_affaire") or "").strip().lower(),
            )

        for row, score in rows_with_score:
            k = _key(row)
            if k not in merged or score > merged[k]["_score"]:
                merged[k] = {**row, "_score": score}

        for row in fts_rows:
            k = _key(row)
            if k not in merged:
                merged[k] = {**row, "_score": 0.0}
            # BM25-scored rows already present — FTS5 doesn't override their score

        if not merged:
            return []

        # ── Confidence bands ──────────────────────────────────────────────────
        max_score = max(v["_score"] for v in merged.values())

        def _confidence(score: float) -> str:
            if max_score == 0:
                return "fts_only"
            pct = score / max_score
            if pct >= 0.60:
                return "high"
            if pct >= 0.25:
                return "medium"
            if score > 0:
                return "low"
            return "fts_only"

        results = sorted(merged.values(), key=lambda r: r["_score"], reverse=True)
        for r in results:
            r["_confidence"] = _confidence(r["_score"])

        logger.info(
            "search_hybrid[%s] %r → %d résultats (top score=%.2f)",
            column, query, len(results[:limit]), max_score,
        )
        return results[:limit]

    def filter_to_column_matches(self, query: str, column: str, results: list[dict]) -> list[dict]:
        """
        Return only rows where the given column's tokens overlap with stemmed query tokens.
        Returns empty list if no row matches (caller keeps original results).
        Uses the same Snowball tokenizer as the BM25 index — handles plurals.
        """
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []
        return [
            row for row in results
            if q_tokens & set(_tokenize(str(row.get(column) or "")))
        ]

    def query_matches_postes(self, query: str, results: list[dict]) -> bool:
        """
        Return True if stemmed query tokens overlap with any nom_poste in results.
        Uses the same Snowball tokenizer as the BM25 index — handles plurals.
        """
        if not results:
            return False
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return bool(results)
        for row in results:
            nom_poste_val = str(row.get("nom_poste") or "")
            np_tokens = set(_tokenize(nom_poste_val))
            if q_tokens & np_tokens:
                return True
        return False

    def get_elements_for_poste(self, nom_poste: str, nom_affaire: str | None = None) -> list[dict]:
        """
        Return all catalog rows (elements/sub-items) for a nom_poste.
        When nom_affaire is provided, restrict to that specific affaire only.
        """
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if nom_affaire:
                rows = conn.execute(
                    """
                    SELECT * FROM catalogue
                    WHERE LOWER(TRIM(nom_poste)) = LOWER(TRIM(?))
                      AND LOWER(TRIM(nom_affaire)) = LOWER(TRIM(?))
                    ORDER BY rowid
                    """,
                    [nom_poste, nom_affaire],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM catalogue WHERE LOWER(TRIM(nom_poste)) = LOWER(TRIM(?)) ORDER BY rowid",
                    [nom_poste],
                ).fetchall()
            return [dict(r) for r in rows]

    def get_current_affaire(self, conversation_id: str) -> str | None:
        """
        Return the nom_affaire locked for this conversation.
        Checks affaire_locks first (set on choice selection), then falls back to panier items.
        """
        if not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as conn:
            # Affaire lock (set before first panier item, via choice card selection)
            row = conn.execute(
                "SELECT nom_affaire FROM devis_affaire_locks WHERE conversation_id = ?",
                [conversation_id],
            ).fetchone()
            if row and row[0]:
                return row[0]
            # Fallback: derive from panier items
            row = conn.execute(
                """
                SELECT nom_affaire FROM devis_paniers
                WHERE conversation_id = ? AND nom_affaire IS NOT NULL AND nom_affaire != ''
                LIMIT 1
                """,
                [conversation_id],
            ).fetchone()
            return row[0] if row else None

    def _find_catalog_row(
        self,
        nom_poste: str,
        nom_affaire: str | None,
        num_poste: str | None,
    ) -> dict | None:
        """
        Find a specific catalog row for enrichment, from most specific to least.
        Priority: (nom_poste + num_poste) > (nom_poste + nom_affaire) > (nom_poste, most recent)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if num_poste:
                row = conn.execute(
                    """
                    SELECT nom_poste, num_poste, ensemble, fournisseur, fourniture,
                           num_affaire, nom_affaire
                    FROM catalogue
                    WHERE LOWER(TRIM(nom_poste)) = LOWER(TRIM(?))
                      AND LOWER(TRIM(num_poste))  = LOWER(TRIM(?))
                    LIMIT 1
                    """,
                    [nom_poste, num_poste],
                ).fetchone()
                if row:
                    return dict(row)

            if nom_affaire:
                row = conn.execute(
                    """
                    SELECT nom_poste, num_poste, ensemble, fournisseur, fourniture,
                           num_affaire, nom_affaire
                    FROM catalogue
                    WHERE LOWER(TRIM(nom_poste))   = LOWER(TRIM(?))
                      AND LOWER(TRIM(nom_affaire)) = LOWER(TRIM(?))
                    ORDER BY CAST(COALESCE(num_affaire, '0') AS INTEGER) DESC
                    LIMIT 1
                    """,
                    [nom_poste, nom_affaire],
                ).fetchone()
                if row:
                    return dict(row)

        return self.get_price(nom_poste)

    def lookup_by_affaire(self, nom_affaire: str, limit: int = 50) -> list[dict]:
        """Return all postes from a given affaire (project name, partial match)."""
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM catalogue
                WHERE  LOWER(nom_affaire) LIKE LOWER(?)
                ORDER  BY num_ensemble, num_poste
                LIMIT  ?
                """,
                [f"%{nom_affaire}%", limit],
            ).fetchall()
            return [dict(r) for r in rows]

    def get_price(self, nom_poste: str) -> dict | None:
        """
        Return price/supplier for a poste using CATALOG_PRICE_STRATEGY.

        Strategy: CATALOG_PRICE_STRATEGY = 'most_recent'
        SQL: ORDER BY CAST(num_affaire AS INTEGER) DESC LIMIT 1
        Hypothesis: higher Num_Affaire → more recent project → more up-to-date price.
        Change CATALOG_PRICE_STRATEGY (and the ORDER BY) to alter this behavior.
        """
        if not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            order = (
                "ORDER BY CAST(COALESCE(num_affaire, '0') AS INTEGER) DESC"
                if CATALOG_PRICE_STRATEGY == "most_recent"
                else ""
            )
            row = conn.execute(
                f"""
                SELECT nom_poste, ensemble, fournisseur, fourniture,
                       num_affaire, nom_affaire
                FROM   catalogue
                WHERE  LOWER(nom_poste) = LOWER(?)
                {order}
                LIMIT  1
                """,
                [nom_poste],
            ).fetchone()
            return dict(row) if row else None

    # ── Panier ─────────────────────────────────────────────────────────────────

    def get_panier(self, conversation_id: str) -> list[dict]:
        """Return all panier items for a conversation (ordered by insertion)."""
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM devis_paniers WHERE conversation_id = ? ORDER BY created_at",
                [conversation_id],
            ).fetchall()
            result = [dict(r) for r in rows]
            for item in result:
                item["is_option"] = bool(item.get("is_option", 0))
            return result

    def add_to_panier(self, conversation_id: str, postes: list[dict]) -> dict:
        """
        Add postes to the panier, enforcing affaire consistency.

        Each poste should carry nom_affaire (and optionally num_poste) from the catalog
        so the exact row can be found. If the panier already has items from an affaire,
        any poste from a different affaire is rejected (returned in "errors").

        Returns {"added": [...], "errors": [...], "current_affaire": str|None}
        """
        now = datetime.now(timezone.utc).isoformat()
        added: list[dict] = []
        errors: list[dict] = []

        # Establish the affaire already in use (if any)
        current_affaire = self.get_current_affaire(conversation_id)

        seen_in_call: set[str] = set()

        # Build set of nom_postes already in the panier (case-insensitive) to prevent duplicates
        existing_panier = self.get_panier(conversation_id)
        already_in_panier: set[str] = {
            (p.get("nom_poste") or "").strip().lower()
            for p in existing_panier
        }

        with sqlite3.connect(self.db_path) as conn:
            for poste in postes:
                # Defensive: LLM occasionally sends a plain string instead of a dict.
                # Only convert if it looks like a real nom_poste (>3 chars), not a stray character.
                if isinstance(poste, str):
                    if len(poste) > 3:
                        poste = {"nom_poste": poste}
                    else:
                        continue
                nom_poste   = (poste.get("nom_poste") or "").strip()
                nom_affaire = (poste.get("nom_affaire") or "").strip() or None
                num_poste   = (poste.get("num_poste") or "").strip() or None

                if not nom_poste:
                    continue

                # Safety: skip duplicate nom_poste within the same add_to_panier call.
                if nom_poste in seen_in_call:
                    logger.info("add_to_panier: doublon ignoré nom_poste=%r", nom_poste)
                    continue
                seen_in_call.add(nom_poste)

                # Safety: skip if already in panier (LLM retry / double add protection)
                if nom_poste.lower() in already_in_panier:
                    logger.info("add_to_panier: déjà dans le panier, ignoré nom_poste=%r", nom_poste)
                    continue

                # Affaire consistency: reject if different from established affaire
                if (
                    current_affaire
                    and nom_affaire
                    and nom_affaire.lower().strip() != current_affaire.lower().strip()
                ):
                    errors.append({
                        "nom_poste": nom_poste,
                        "error": "affaire_mismatch",
                        "current_affaire": current_affaire,
                        "requested_affaire": nom_affaire,
                    })
                    continue

                # Find specific catalog row for enrichment
                price_data = self._find_catalog_row(nom_poste, nom_affaire, num_poste) or {}

                # Confirm affaire from catalog row (more reliable than LLM-supplied value)
                confirmed_affaire = (
                    price_data.get("nom_affaire")
                    or nom_affaire
                    or current_affaire
                )
                if not current_affaire:
                    current_affaire = confirmed_affaire

                item: dict = {
                    "id": str(uuid.uuid4()),
                    "conversation_id": conversation_id,
                    "nom_poste": nom_poste,
                    "num_poste": num_poste or str(price_data.get("num_poste") or ""),
                    "ensemble": poste.get("ensemble") or price_data.get("ensemble"),
                    "quantite": int(poste.get("quantite", 1)),
                    "fournisseur": price_data.get("fournisseur"),
                    "fourniture": price_data.get("fourniture"),
                    "num_affaire": str(price_data.get("num_affaire") or ""),
                    "nom_affaire": confirmed_affaire,
                    "nbre_jours_etude":   int(poste.get("nbre_jours_etude", 0) or 0),
                    "nbre_jours_atelier": int(poste.get("nbre_jours_atelier", 0) or 0),
                    "nbre_jours_client":  int(poste.get("nbre_jours_client", 0) or 0),
                    "is_option":          bool(poste.get("is_option", False)),
                    "created_at": now,
                }
                conn.execute(
                    """
                    INSERT INTO devis_paniers
                      (id, conversation_id, nom_poste, num_poste, ensemble, quantite,
                       fournisseur, fourniture, num_affaire, nom_affaire, item_type, created_at,
                       nbre_jours_etude, nbre_jours_atelier, nbre_jours_client, is_option)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        item["id"], item["conversation_id"], item["nom_poste"],
                        item["num_poste"], item["ensemble"], item["quantite"],
                        item["fournisseur"], item["fourniture"],
                        item["num_affaire"], item["nom_affaire"], "poste", item["created_at"],
                        item["nbre_jours_etude"], item["nbre_jours_atelier"],
                        item["nbre_jours_client"], int(item["is_option"]),
                    ],
                )
                added.append(item)
            conn.commit()

        return {"added": added, "errors": errors, "current_affaire": current_affaire}

    def get_postes_aggregated(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """
        Given a list of (num_poste, nom_affaire) pairs, return one aggregated row
        per pair: prix_total = SUM(fourniture), nb_elements = COUNT(*).
        Uses SQL GROUP BY for deterministic, NULL-safe aggregation.
        Pairs should be pre-normalized: num_poste stripped, nom_affaire lowercased+stripped.
        """
        if not pairs or not self.db_path.exists():
            return []
        values_sql = ",".join("(?,?)" for _ in pairs)
        flat_params = [v for p in pairs for v in p]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                WITH pairs(np, na) AS (VALUES {values_sql})
                SELECT c.nom_poste, c.nom_affaire, c.num_poste, c.num_affaire,
                       c.num_ensemble, c.ensemble,
                       ROUND(SUM(CAST(COALESCE(c.fourniture, 0) AS REAL)), 2) AS prix_total,
                       COUNT(*) AS nb_elements
                FROM   catalogue c
                JOIN   pairs ON TRIM(COALESCE(c.num_poste, ''))  = pairs.np
                            AND LOWER(TRIM(c.nom_affaire))        = pairs.na
                GROUP  BY c.num_poste, c.nom_affaire, c.nom_poste
                """,
                flat_params,
            ).fetchall()
            return [dict(r) for r in rows]

    def get_postes_aggregated_by_name(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """
        Like get_postes_aggregated but keyed by (nom_poste, nom_affaire) instead of
        (num_poste, nom_affaire). Safer because num_poste is not always unique per
        poste within an affaire.
        Pairs should be pre-normalized: nom_poste stripped, nom_affaire lowercased+stripped.
        """
        if not pairs or not self.db_path.exists():
            return []
        values_sql = ",".join("(?,?)" for _ in pairs)
        flat_params = [v for p in pairs for v in p]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                WITH pairs(np, na) AS (VALUES {values_sql})
                SELECT c.nom_poste, c.nom_affaire, c.num_poste, c.num_affaire,
                       c.num_ensemble, c.ensemble,
                       ROUND(SUM(CAST(COALESCE(c.fourniture, 0) AS REAL)), 2) AS prix_total,
                       COUNT(*) AS nb_elements
                FROM   catalogue c
                JOIN   pairs ON LOWER(TRIM(c.nom_poste))   = LOWER(pairs.np)
                            AND LOWER(TRIM(c.nom_affaire))  = pairs.na
                GROUP  BY c.nom_poste, c.nom_affaire
                """,
                flat_params,
            ).fetchall()
            return [dict(r) for r in rows]

    def add_element_to_panier(self, conversation_id: str, element_data: dict) -> dict:
        """
        Add a catalog element (col H sub-component) directly to the panier
        as an independent line item (item_type='element').
        element_data keys: elements, nom_poste, nom_affaire, fournisseur,
                           fourniture, ensemble, num_affaire, num_poste
        """
        nom_element = (element_data.get("elements") or "").strip()
        if not nom_element:
            return {"added": [], "errors": [{"error": "nom_element vide"}]}

        now = datetime.now(timezone.utc).isoformat()
        item = {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "nom_poste": nom_element,          # reuse nom_poste field for display
            "num_poste": (element_data.get("num_poste") or "").strip() or None,
            "ensemble": (element_data.get("ensemble") or "").strip() or None,
            "quantite": 1,
            "fournisseur": (element_data.get("fournisseur") or "").strip() or None,
            "fourniture": (element_data.get("fourniture") or "").strip() or None,
            "num_affaire": (element_data.get("num_affaire") or "").strip() or None,
            "nom_affaire": (element_data.get("nom_affaire") or "").strip() or None,
            "item_type": "element",
            "created_at": now,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO devis_paniers
                  (id, conversation_id, nom_poste, num_poste, ensemble, quantite,
                   fournisseur, fourniture, num_affaire, nom_affaire, item_type, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    item["id"], item["conversation_id"], item["nom_poste"],
                    item["num_poste"], item["ensemble"], item["quantite"],
                    item["fournisseur"], item["fourniture"],
                    item["num_affaire"], item["nom_affaire"],
                    item["item_type"], item["created_at"],
                ],
            )
            conn.commit()
        return {"added": [item], "errors": []}

    def clear_panier(self, conversation_id: str) -> None:
        """Remove all panier items and affaire lock for a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM devis_paniers WHERE conversation_id = ?",
                [conversation_id],
            )
            conn.execute(
                "DELETE FROM devis_affaire_locks WHERE conversation_id = ?",
                [conversation_id],
            )
            conn.commit()

    def remove_from_panier(self, conversation_id: str, item_id: str) -> None:
        """Remove a specific item from the panier."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM devis_paniers WHERE conversation_id = ? AND id = ?",
                [conversation_id, item_id],
            )
            conn.commit()

    def update_panier_item(self, conversation_id: str, item_id: str, fields: dict) -> dict:
        """Partial update. Allowed keys: nbre_jours_etude, nbre_jours_atelier, nbre_jours_client, is_option.
        Validates nbre_jours_* >= 0. Raises ValueError if item not found or doesn't belong to conversation.
        Returns updated item dict."""
        allowed = {"nbre_jours_etude", "nbre_jours_atelier", "nbre_jours_client", "is_option"}
        update_fields = {k: v for k, v in fields.items() if k in allowed}
        if not update_fields:
            raise ValueError("No updatable fields provided")
        for key in ("nbre_jours_etude", "nbre_jours_atelier", "nbre_jours_client"):
            if key in update_fields and update_fields[key] is not None and int(update_fields[key]) < 0:
                raise ValueError(f"{key} must be >= 0")
        if "is_option" in update_fields:
            update_fields["is_option"] = int(bool(update_fields["is_option"]))
        set_clause = ", ".join(f"{k} = ?" for k in update_fields)
        values = list(update_fields.values()) + [conversation_id, item_id]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE devis_paniers SET {set_clause} WHERE conversation_id = ? AND id = ?",
                values,
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Item {item_id} not found in conversation {conversation_id}")
            conn.commit()
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM devis_paniers WHERE id = ?", [item_id]).fetchone()
        if row is None:
            raise ValueError(f"Item {item_id} not found after update")
        item = dict(row)
        item["is_option"] = bool(item.get("is_option", 0))
        return item

    def get_devis_settings(self, conversation_id: str) -> dict:
        """Return {"coefficient": float, "coef_final": float}. Returns defaults if no row exists."""
        defaults = {"coefficient": 0.0, "coef_final": 0.0}
        if not self.db_path.exists():
            return defaults
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT coefficient, coef_final FROM devis_settings WHERE conversation_id = ?",
                    [conversation_id],
                ).fetchone()
                return dict(row) if row else defaults
        except Exception:
            return defaults

    def set_devis_settings(self, conversation_id: str, coefficient: float, coef_final: float) -> None:
        """INSERT OR REPLACE into devis_settings."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO devis_settings (conversation_id, coefficient, coef_final)
                VALUES (?, ?, ?)
                """,
                [conversation_id, coefficient, coef_final],
            )
            conn.commit()

    # ── Status ─────────────────────────────────────────────────────────────────

    def get_all_postes(self) -> list[str]:
        """Return all distinct nom_poste values from the catalog (for LLM cross-referencing)."""
        if not self.db_path.exists():
            return []
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT nom_poste FROM catalogue "
                    "WHERE nom_poste IS NOT NULL AND TRIM(nom_poste) != '' "
                    "ORDER BY nom_poste"
                ).fetchall()
                return [r[0] for r in rows]
        except Exception:
            return []

    @property
    def status(self) -> dict:
        if not self.db_path.exists():
            return {"loaded": False, "rows": 0, "unique_postes": 0, "columns": []}
        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM catalogue").fetchone()[0]
                unique_postes = conn.execute(
                    "SELECT COUNT(DISTINCT nom_poste) FROM catalogue "
                    "WHERE nom_poste IS NOT NULL AND TRIM(nom_poste) != ''"
                ).fetchone()[0]
                cols = [
                    r[1]
                    for r in conn.execute("PRAGMA table_info(catalogue)").fetchall()
                ]
                return {"loaded": True, "rows": count, "unique_postes": unique_postes, "columns": cols}
        except Exception:
            return {"loaded": False, "rows": 0, "unique_postes": 0, "columns": []}
