"""
core/bm25_utils.py — Utilitaires BM25 partagés (Snowball FR).

Utilisé par :
  - backend/adapters/catalog_adapter.py (catalogue devis)
  - backend/adapters/excel_collection_adapter.py (Excel RAG SQL)
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

_stemmer: Any = None


def get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem.snowball import FrenchStemmer
            _stemmer = FrenchStemmer()
        except Exception as exc:
            logger.warning("Stemmer NLTK indisponible (%s) — fallback tokenisation brute", exc)
    return _stemmer


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = get_stemmer()
    if stemmer is None:
        return tokens
    return [stemmer.stem(t) for t in tokens]


def tokenize(text: str) -> list[str]:
    return stem_tokens(re.findall(r"\w+", text.lower()))


class BM25Index:
    """
    Index BM25 par colonne sur un ensemble de lignes dict.

    Usage :
        idx = BM25Index(rows, text_columns=["nom", "description"])
        results = idx.search("moteur convoyeur", column="nom", limit=10)
    """

    def __init__(self, rows: list[dict], text_columns: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self._rows = rows
        self._text_columns = text_columns
        self._indexes: dict[str, Any] = {}

        for col in text_columns:
            texts = [str(r.get(col) or "") for r in rows]
            tokenized = [tokenize(t) for t in texts]
            self._indexes[col] = BM25Okapi(tokenized)

    def search(self, query: str, column: str | None = None, limit: int = 10) -> list[dict]:
        """
        Recherche BM25 dans la colonne spécifiée (ou la première colonne si None).
        Retourne les lignes triées par score décroissant (score > 0 uniquement).
        """
        if not self._indexes:
            return []

        col = column if column in self._indexes else self._text_columns[0]
        bm25 = self._indexes[col]
        tokens = tokenize(query)
        scores = bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked[:limit]:
            if scores[idx] > 0:
                results.append(self._rows[idx])
        return results

    def search_all_columns(self, query: str, limit: int = 10) -> list[dict]:
        """
        Recherche sur toutes les colonnes texte, fusion par score max par ligne.
        Utile quand on ne sait pas dans quelle colonne chercher.
        """
        from rank_bm25 import BM25Okapi

        tokens = tokenize(query)
        combined: dict[int, float] = {}

        for col, bm25 in self._indexes.items():
            scores = bm25.get_scores(tokens)
            for i, score in enumerate(scores):
                combined[i] = max(combined.get(i, 0.0), score)

        ranked = sorted(combined.keys(), key=lambda i: combined[i], reverse=True)
        return [self._rows[idx] for idx in ranked[:limit] if combined[idx] > 0]
