"""
adapters/excel_collection_adapter.py — Pipeline SQL déterministe pour fichiers Excel.

Quand un Excel est uploadé dans une collection RAG, il est indexé EN PARALLÈLE :
  1. Pipeline vecteur (existant) — parsers.py → Qdrant/Chroma (non modifié)
  2. Pipeline SQL (ce fichier)   — SQLite dynamique + NL2SQL + BM25 fallback

Le endpoint /collections/{name}/excel/compare permet de comparer les deux pipelines
côte à côte pour valider lequel donne de meilleurs résultats.

Catalogue devis (catalog_adapter.py) : NON MODIFIÉ — pipeline séparé.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Config Ollama (même variables d'env que le reste du backend) ───────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# ── Chemin SQLite dédié (séparé de catalogue.db) ──────────────────────────────
_DEFAULT_DB_PATH = Path(os.environ.get("EXCEL_AUTO_DB_PATH", "/app/documents/excel_collections.db"))

# ── Prompt NL2SQL ─────────────────────────────────────────────────────────────
_NL2SQL_PROMPT = """\
Tu es un expert SQLite. Génère une requête SQL SELECT pour répondre à la question suivante.

Tables disponibles dans la base :
{tables_description}

Question : {question}

Règles STRICTES :
- Retourne UNIQUEMENT le SQL brut, sans markdown, sans explication, sans commentaire
- Une seule instruction SELECT
- SQL compatible SQLite (pas de fonctions MySQL/PostgreSQL)
- Si la question est vague ou générale : SELECT * FROM <table_la_plus_pertinente> LIMIT 20
- Ne génère JAMAIS de DROP, DELETE, UPDATE, INSERT

RECHERCHE TEXTE — UTILISE FTS5 PAR DÉFAUT (insensible aux accents ET à la casse) :
    SELECT t.* FROM "<table>" t
    JOIN "<table>_fts" ON "<table>_fts".row_id = t.rowid
    WHERE "<table>_fts" MATCH 'mot_sans_accent'
- CRITIQUE : NOM COMPLET de la table FTS dans le WHERE MATCH, jamais un alias
- Exemple : WHERE "xl_79a50102_d_tail_prix_fts" MATCH 'mecanique'
- Exemple : pour chercher "Mécanique" → MATCH 'mecanique' (pas besoin d'accent)
- Pour plusieurs mots : MATCH 'mecanique poste' ou MATCH 'mecanique OR meca'
- LIKE uniquement pour colonnes numériques ou comparaisons exactes
"""


def _norm(name: str) -> str:
    """Normalise un nom de colonne → snake_case compatible SQLite."""
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9_]", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    if not n or n[0].isdigit():
        n = "col_" + n
    return n or "col"


def _slug(name: str, maxlen: int = 20) -> str:
    """Génère un slug court pour les noms de tables SQLite."""
    s = re.sub(r"[^a-z0-9]", "_", name.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] or "sheet"


def _sha8(text: str) -> str:
    """Hash SHA256 tronqué à 8 caractères pour unicité."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _detect_column_type(series) -> str:
    """Détecte le type SQLite d'une colonne pandas."""
    import pandas as pd
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


class ExcelCollectionAdapter:
    """
    Adapter pour indexer des fichiers Excel dans SQLite et les interroger via NL2SQL.

    Usage :
        adapter = ExcelCollectionAdapter()
        adapter.init_db()
        result = adapter.ingest(Path("tarifs.xlsx"), "ma_collection")
        answer = adapter.nl2sql_query("prix du moteur 3kW", "ma_collection")
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        self.db_path = db_path

    # ── Initialisation ────────────────────────────────────────────────────────

    def init_db(self) -> None:
        """Crée le registry excel_collections (idempotent)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS excel_collections (
                    id           TEXT PRIMARY KEY,
                    collection   TEXT NOT NULL,
                    filename     TEXT NOT NULL,
                    sheet_name   TEXT NOT NULL,
                    table_name   TEXT NOT NULL,
                    columns_json TEXT NOT NULL,
                    sample_rows  TEXT NOT NULL,
                    row_count    INTEGER DEFAULT 0,
                    uploaded_at  TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ec_collection "
                "ON excel_collections(collection)"
            )
            conn.commit()
        logger.info("ExcelCollectionAdapter : DB initialisée → %s", self.db_path)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, excel_path: Path, collection_name: str) -> dict:
        """
        Charge toutes les feuilles visibles d'un Excel dans SQLite.

        Pour chaque feuille :
          - Crée une table xl_<hash>_<sheet_slug> (DROP + RECREATE si déjà présente)
          - Construit un index FTS5 sur les colonnes texte
          - Enregistre dans excel_collections

        Retourne : {collection, filename, tables: [{table_name, sheet, rows, columns}]}
        """
        import pandas as pd

        filename = excel_path.name
        file_hash = _sha8(filename + collection_name)

        # Feuilles visibles seulement (.xls via xlrd)
        hidden_sheets: set[str] = set()
        if excel_path.suffix.lower() == ".xls":
            try:
                import xlrd
                wb = xlrd.open_workbook(str(excel_path))
                hidden_sheets = {
                    wb.sheet_names()[i]
                    for i in range(wb.nsheets)
                    if wb.sheet_visibility(i) != 0
                }
            except Exception as e:
                logger.debug("xlrd visibility check échoué (%s)", e)

        excel_file = pd.ExcelFile(str(excel_path))
        visible_sheets = [s for s in excel_file.sheet_names if s not in hidden_sheets]

        tables_info = []

        with sqlite3.connect(self.db_path) as conn:
            for sheet_name in visible_sheets:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if df.empty:
                    logger.debug("Feuille '%s' vide — ignorée", sheet_name)
                    continue

                # Normalise les colonnes
                original_cols = df.columns.tolist()
                norm_cols = [_norm(str(c)) for c in original_cols]
                # Déduplique si collision après normalisation
                seen: dict[str, int] = {}
                deduped = []
                for col in norm_cols:
                    if col in seen:
                        seen[col] += 1
                        deduped.append(f"{col}_{seen[col]}")
                    else:
                        seen[col] = 0
                        deduped.append(col)
                df.columns = deduped

                table_name = f"xl_{file_hash}_{_slug(sheet_name)}"

                # Colonnes avec types
                col_defs = []
                col_info = []
                for col in df.columns:
                    dtype = _detect_column_type(df[col])
                    col_defs.append(f'"{col}" {dtype}')
                    col_info.append({"name": col, "type": dtype})

                # Drop + recreate table
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}_fts"')
                conn.execute(
                    f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'
                )

                # Insert données (NaN → NULL)
                placeholders = ", ".join("?" * len(df.columns))
                for _, row in df.iterrows():
                    values = []
                    for v in row:
                        try:
                            import math
                            if pd.isna(v):
                                values.append(None)
                            else:
                                values.append(v)
                        except (TypeError, ValueError):
                            values.append(v)
                    conn.execute(
                        f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                        values,
                    )

                # FTS5 sur les colonnes texte (unicode61 = insensible accents + casse)
                text_cols = [c["name"] for c in col_info if c["type"] == "TEXT"]
                if text_cols:
                    fts_cols = ", ".join(f'"{c}"' for c in text_cols)
                    conn.execute(
                        f'CREATE VIRTUAL TABLE "{table_name}_fts" '
                        f'USING fts5(row_id UNINDEXED, {fts_cols}, '
                        f'tokenize="unicode61 remove_diacritics 2")'
                    )
                    coalesced = ", ".join(f'COALESCE("{c}", "")' for c in text_cols)
                    conn.execute(
                        f'INSERT INTO "{table_name}_fts"(row_id, {fts_cols}) '
                        f'SELECT rowid, {coalesced} FROM "{table_name}"'
                    )

                # Échantillon de 10 lignes pour le prompt NL2SQL (structure hiérarchique visible)
                sample_rows = df.head(10).fillna("").to_dict(orient="records")
                # Convertir les types non-sérialisables
                sample_rows = _serialize_rows(sample_rows)

                # Supprimer l'ancienne entrée si elle existe
                conn.execute(
                    "DELETE FROM excel_collections "
                    "WHERE collection = ? AND filename = ? AND sheet_name = ?",
                    [collection_name, filename, sheet_name],
                )

                # Enregistrer dans le registry
                conn.execute(
                    """
                    INSERT INTO excel_collections
                      (id, collection, filename, sheet_name, table_name,
                       columns_json, sample_rows, row_count, uploaded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        collection_name,
                        filename,
                        sheet_name,
                        table_name,
                        json.dumps(col_info, ensure_ascii=False),
                        json.dumps(sample_rows, ensure_ascii=False, default=str),
                        len(df),
                        datetime.now(timezone.utc).isoformat(),
                    ],
                )

                tables_info.append({
                    "table_name": table_name,
                    "sheet": sheet_name,
                    "rows": len(df),
                    "columns": col_info,
                })

                logger.info(
                    "Excel ingest SQL : collection=%s fichier=%s feuille=%s "
                    "→ table=%s (%d lignes, %d colonnes)",
                    collection_name, filename, sheet_name,
                    table_name, len(df), len(col_info),
                )

            conn.commit()

        return {
            "collection": collection_name,
            "filename": filename,
            "tables": tables_info,
        }

    # ── Requêtes ──────────────────────────────────────────────────────────────

    def has_excel_data(self, collection_name: str) -> bool:
        """True si la collection a au moins une table Excel indexée."""
        if not self.db_path.exists():
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT 1 FROM excel_collections WHERE collection = ? LIMIT 1",
                    [collection_name],
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def list_tables(self, collection_name: str) -> list[dict]:
        """Retourne les tables Excel de la collection avec schéma + échantillon."""
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, filename, sheet_name, table_name,
                       columns_json, sample_rows, row_count, uploaded_at
                FROM excel_collections
                WHERE collection = ?
                ORDER BY uploaded_at
                """,
                [collection_name],
            ).fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r["id"],
                "filename": r["filename"],
                "sheet_name": r["sheet_name"],
                "table_name": r["table_name"],
                "columns": json.loads(r["columns_json"]),
                "sample_rows": json.loads(r["sample_rows"]),
                "row_count": r["row_count"],
                "uploaded_at": r["uploaded_at"],
            })
        return result

    def list_all_collections(self) -> list[str]:
        """Retourne les noms de collections ayant des données Excel."""
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT collection FROM excel_collections ORDER BY collection"
            ).fetchall()
        return [r[0] for r in rows]

    def nl2sql_query(self, question: str, collection_name: str, filename: str | None = None) -> dict:
        """
        Interroge les Excel de la collection via NL2SQL déterministe.

        Pipeline :
          1. Lire le schéma des tables de la collection (+ 10 lignes exemple)
          2. Appel Ollama temperature=0 → génère SQL
          3. Exécuter le SQL → résultats exacts
          4. Fallback BM25 si SQL vide ou erreur

        filename : si renseigné, restreint aux tables de ce fichier uniquement.
        """
        tables = self.list_tables(collection_name)
        if filename:
            tables = [t for t in tables if t["filename"] == filename]
        if not tables:
            return {"method": "empty", "sql": "", "results": [], "error": "Aucune table Excel pour cette collection"}

        # ── Construire la description des tables pour le prompt ────────────────
        tables_desc = _build_tables_description(tables, self.db_path)
        prompt = _NL2SQL_PROMPT.format(
            tables_description=tables_desc,
            question=question,
        )

        # ── Générer le SQL via Ollama ──────────────────────────────────────────
        generated_sql = ""
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 300, "num_ctx": 8192},  # aligné — évite reload KV cache
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            generated_sql = _extract_sql(raw)
            logger.info("NL2SQL généré : %s", generated_sql[:200])
        except Exception as e:
            logger.warning("NL2SQL Ollama échoué (%s) — fallback BM25", e)
            return self._bm25_result(question, collection_name, sql_error=str(e), filename=filename)

        if not generated_sql:
            logger.warning("NL2SQL : SQL vide généré — fallback BM25")
            return self._bm25_result(question, collection_name, sql_error="SQL vide généré", filename=filename)

        # ── Exécuter le SQL ────────────────────────────────────────────────────
        try:
            results = self._execute_sql(generated_sql)
            if results:
                logger.info("NL2SQL SQL → %d résultats", len(results))
                return {"method": "sql", "sql": generated_sql, "results": results}
            else:
                logger.info("NL2SQL SQL → 0 résultat — fallback BM25")
                return self._bm25_result(question, collection_name, sql=generated_sql, filename=filename)
        except Exception as e:
            logger.warning("NL2SQL exécution échouée (%s) — fallback BM25", e)
            return self._bm25_result(question, collection_name, sql=generated_sql, sql_error=str(e), filename=filename)

    def _execute_sql(self, sql: str) -> list[dict]:
        """Exécute un SELECT dans la DB et retourne les résultats."""
        # Sécurité : n'autoriser que les SELECT
        sql_clean = sql.strip().upper()
        if not sql_clean.startswith("SELECT"):
            raise ValueError(f"Seuls les SELECT sont autorisés (reçu : {sql[:50]})")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def _bm25_result(
        self,
        query: str,
        collection_name: str,
        sql: str = "",
        sql_error: str = "",
        filename: str | None = None,
    ) -> dict:
        """Fallback BM25 quand le SQL ne donne rien ou échoue."""
        results = self.bm25_search(query, collection_name, filename=filename)
        out = {"method": "bm25", "sql": sql, "results": results}
        if sql_error:
            out["error"] = sql_error
        return out

    def bm25_search(self, query: str, collection_name: str, limit: int = 10, filename: str | None = None) -> list[dict]:
        """
        Recherche BM25 Snowball FR sur toutes les tables texte de la collection.
        Utilisé comme fallback quand le SQL ne retourne rien.
        filename : si renseigné, restreint la recherche à ce fichier uniquement.
        """
        from core.bm25_utils import BM25Index, tokenize

        tables = self.list_tables(collection_name)
        if filename:
            tables = [t for t in tables if t["filename"] == filename]
        if not tables:
            return []

        all_results: list[dict] = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for table_info in tables:
                table_name = table_info["table_name"]
                text_cols = [
                    c["name"] for c in table_info["columns"]
                    if c["type"] == "TEXT"
                ]
                if not text_cols:
                    continue

                try:
                    rows = [
                        dict(r)
                        for r in conn.execute(f'SELECT * FROM "{table_name}"').fetchall()
                    ]
                except Exception as e:
                    logger.warning("BM25 : lecture table %s échouée (%s)", table_name, e)
                    continue

                if not rows:
                    continue

                idx = BM25Index(rows, text_cols)
                results = idx.search_all_columns(query, limit=limit)
                for r in results:
                    r["_source_file"] = table_info["filename"]
                    r["_source_sheet"] = table_info["sheet_name"]
                all_results.extend(results)

        return all_results[:limit]

    # ── Suppression ───────────────────────────────────────────────────────────

    def delete_file(self, collection_name: str, filename: str) -> bool:
        """Supprime toutes les tables d'un fichier Excel de la collection."""
        tables = self.list_tables(collection_name)
        file_tables = [t for t in tables if t["filename"] == filename]
        if not file_tables:
            return False

        with sqlite3.connect(self.db_path) as conn:
            for t in file_tables:
                tn = t["table_name"]
                conn.execute(f'DROP TABLE IF EXISTS "{tn}"')
                conn.execute(f'DROP TABLE IF EXISTS "{tn}_fts"')
                logger.info("Suppression table %s", tn)
            conn.execute(
                "DELETE FROM excel_collections "
                "WHERE collection = ? AND filename = ?",
                [collection_name, filename],
            )
            conn.commit()
        return True

    def delete_collection_data(self, collection_name: str) -> int:
        """Supprime toutes les tables Excel d'une collection. Retourne le nb de tables supprimées."""
        tables = self.list_tables(collection_name)
        with sqlite3.connect(self.db_path) as conn:
            for t in tables:
                tn = t["table_name"]
                conn.execute(f'DROP TABLE IF EXISTS "{tn}"')
                conn.execute(f'DROP TABLE IF EXISTS "{tn}_fts"')
            conn.execute(
                "DELETE FROM excel_collections WHERE collection = ?",
                [collection_name],
            )
            conn.commit()
        return len(tables)


# ── Helpers privés ────────────────────────────────────────────────────────────

_CATEGORICAL_THRESHOLD = 30  # colonnes TEXT avec < N valeurs distinctes → afficher toutes les valeurs


def _build_tables_description(tables: list[dict], db_path: Path) -> str:
    """
    Construit la description des tables pour le prompt NL2SQL.

    Pour chaque colonne TEXT :
    - < 30 valeurs distinctes → catégorielle → affiche toutes les valeurs possibles
    - ≥ 30 valeurs distinctes → texte libre → affiche 3 exemples seulement

    Cela permet au LLM de générer WHERE col = 'ValeurExacte' plutôt que FTS5 MATCH.
    """
    parts = []
    with sqlite3.connect(db_path) as conn:
        for t in tables:
            col_lines = []
            for col in t["columns"]:
                name, ctype = col["name"], col["type"]
                if ctype != "TEXT":
                    col_lines.append(f'  {name} ({ctype})')
                    continue
                try:
                    count = conn.execute(
                        f'SELECT COUNT(DISTINCT "{name}") FROM "{t["table_name"]}" '
                        f'WHERE "{name}" IS NOT NULL AND "{name}" != ""'
                    ).fetchone()[0]

                    if 0 < count <= _CATEGORICAL_THRESHOLD:
                        vals = conn.execute(
                            f'SELECT DISTINCT "{name}" FROM "{t["table_name"]}" '
                            f'WHERE "{name}" IS NOT NULL AND "{name}" != "" ORDER BY "{name}"'
                        ).fetchall()
                        vals_str = ", ".join(f'"{r[0]}"' for r in vals)
                        col_lines.append(f'  {name} TEXT  → valeurs: {vals_str}')
                    else:
                        ex = conn.execute(
                            f'SELECT DISTINCT "{name}" FROM "{t["table_name"]}" '
                            f'WHERE "{name}" IS NOT NULL AND "{name}" != "" LIMIT 3'
                        ).fetchall()
                        ex_str = ", ".join(f'"{r[0]}"' for r in ex)
                        col_lines.append(f'  {name} TEXT  → ex: {ex_str}… ({count} valeurs)')
                except Exception:
                    col_lines.append(f'  {name} TEXT')

            parts.append(
                f"Table : {t['table_name']}\n"
                f"  Fichier : {t['filename']} / Feuille : {t['sheet_name']}\n"
                f"  {t['row_count']} lignes\n"
                + "\n".join(col_lines)
            )
    return "\n\n".join(parts)


def _extract_sql(raw: str) -> str:
    """
    Extrait le SQL d'une réponse Ollama.
    Gère les cas où le LLM ajoute du markdown (```sql ... ```) malgré les instructions.
    """
    # Enlever les balises markdown SQL
    m = re.search(r"```(?:sql)?\s*(SELECT.+?)```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Chercher directement un SELECT
    m = re.search(r"(SELECT\s+.+)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
        # Tronquer après le premier ; si présent
        if ";" in sql:
            sql = sql[: sql.index(";") + 1]
        return sql

    return ""


def _serialize_rows(rows: list[dict]) -> list[dict]:
    """Convertit les types pandas non-JSON-sérialisables en types Python natifs."""
    import math
    result = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            elif hasattr(v, "item"):  # numpy scalars
                clean[k] = v.item()
            else:
                clean[k] = v
        result.append(clean)
    return result
