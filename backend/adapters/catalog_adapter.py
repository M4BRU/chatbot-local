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


# ── In-memory BM25 catalog index ───────────────────────────────────────────────

class _CatalogBM25Index:
    """BM25 index over catalog rows for stemmed full-text search."""

    def __init__(self, rows: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self._rows = rows
        texts = [
            " ".join(
                str(r.get(c) or "")
                for c in _FTS_COLS
            )
            for r in rows
        ]
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Relative threshold: only keep results with score >= 30% of the best score.
        # This prevents weakly-related items (score near 0) from appearing in results.
        max_score = scores[ranked[0]] if ranked else 0
        min_threshold = max_score * 0.50 if max_score > 0 else 0
        results = []
        for idx in ranked[:limit]:
            if scores[idx] > 0 and scores[idx] >= min_threshold:
                results.append(self._rows[idx])
        return results


class CatalogAdapter:
    def __init__(self, db_path: Path = CATALOG_DB_PATH) -> None:
        self.db_path = db_path
        self._bm25_index: _CatalogBM25Index | None = None

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
            # Migrations: add columns if upgrading from older schema
            for _migration in [
                "ALTER TABLE devis_paniers ADD COLUMN num_poste TEXT",
                "ALTER TABLE devis_paniers ADD COLUMN item_type TEXT NOT NULL DEFAULT 'poste'",
                "ALTER TABLE devis_affaire_locks ADD COLUMN search_all INTEGER NOT NULL DEFAULT 0",
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

        # Build in-memory BM25 index (handles plurals via Snowball stemmer)
        rows = df.to_dict(orient="records")
        try:
            self._bm25_index = _CatalogBM25Index(rows)
            logger.info("BM25 catalogue : index construit (%d lignes)", len(rows))
        except Exception as exc:
            self._bm25_index = None
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
        """Rebuild BM25 index from SQLite if not in memory (e.g. after server restart)."""
        if self._bm25_index is not None or not self.db_path.exists():
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = [dict(r) for r in conn.execute("SELECT * FROM catalogue").fetchall()]
            if rows:
                self._bm25_index = _CatalogBM25Index(rows)
                logger.info("BM25 catalogue : index reconstruit depuis SQLite (%d lignes)", len(rows))
        except Exception as exc:
            logger.warning("BM25 catalogue : reconstruction échouée (%s)", exc)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search catalog rows using BM25 (Snowball-stemmed, handles plurals).
        Falls back to SQLite FTS5, then LIKE if BM25 index is unavailable.
        """
        if not self.db_path.exists():
            return []

        # ── Priority 1: BM25 (stemmed — handles plurals like orbiteurs/orbiteur) ──
        self._rebuild_bm25_if_needed()
        if self._bm25_index is not None:
            results = self._bm25_index.search(query, limit=limit)
            if results:
                logger.debug("search_catalog BM25 → %d résultats pour %r", len(results), query)
                return results
            # No BM25 hits (all scores = 0) → fall through to FTS5/LIKE
            logger.debug("search_catalog BM25 score=0 pour %r — fallback FTS5", query)

        # ── Priority 2: FTS5 (exact token match, no stemming) ─────────────────
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            fts_ok = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='catalogue_fts'"
            ).fetchone()

            if fts_ok:
                try:
                    rows = conn.execute(
                        """
                        SELECT c.*
                        FROM   catalogue c
                        JOIN   catalogue_fts fts ON c.rowid = fts.row_id
                        WHERE  catalogue_fts MATCH ?
                        ORDER  BY rank
                        LIMIT  ?
                        """,
                        [query, limit],
                    ).fetchall()
                    if rows:
                        return [dict(r) for r in rows]
                except sqlite3.OperationalError:
                    pass

            # ── Priority 3: LIKE fallback ──────────────────────────────────────
            pattern = f"%{query}%"
            available = [
                r[1]
                for r in conn.execute("PRAGMA table_info(catalogue)").fetchall()
                if r[1] in _FTS_COLS
            ]
            if not available:
                return []

            conditions = " OR ".join(f'"{c}" LIKE ?' for c in available)
            rows = conn.execute(
                f"SELECT * FROM catalogue WHERE {conditions} LIMIT ?",
                [pattern] * len(available) + [limit],
            ).fetchall()
            return [dict(r) for r in rows]

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
            return [dict(r) for r in rows]

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
                # The catalog has multiple rows per poste (one per element/sub-component).
                # If the LLM passes the same nom_poste multiple times, keep only the first.
                if nom_poste in seen_in_call:
                    logger.info("add_to_panier: doublon ignoré nom_poste=%r", nom_poste)
                    continue
                seen_in_call.add(nom_poste)

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
                    "created_at": now,
                }
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
                        item["num_affaire"], item["nom_affaire"], "poste", item["created_at"],
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
