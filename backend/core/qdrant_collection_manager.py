"""
core/qdrant_collection_manager.py — Gestion multi-collections Qdrant via HTTP.

Remplace CollectionManager (ChromaDB) quand VECTOR_DB=qdrant.
Interface identique : collection_existe(), creer_collection(), get_collection(),
lister_collections(), supprimer_collection().

Modes :
  EMBED_SPARSE=false (défaut) : dense uniquement (cosine, 1024-dim)
  EMBED_SPARSE=true           : hybrid dense+sparse (bge-m3 SPLADE natif Qdrant)

La collection est créée automatiquement au premier add_texts() par langchain-qdrant
(dimensions inférées depuis le premier vecteur → pas besoin de hardcoder EMBED_DIM).

Notes :
  - BM25 désactivé côté search.py quand VECTOR_DB=qdrant (Qdrant HYBRID le remplace)
  - keyword_fallback, tail_extension, section_retrieval désactivés (where_document absent)
  - Vérifier VECTOR_DB=qdrant dans docker-compose.yml pour activer ce manager
"""

import logging
import os

logger = logging.getLogger(__name__)

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
EMBED_SPARSE = os.environ.get("EMBED_SPARSE", "false").lower() == "true"


def _convertir_filtre(filtre: dict | None):
    """
    Convertit un filtre ChromaDB vers un filtre Qdrant.

    ChromaDB : {"machine": {"$eq": "GEMINI"}}
               {"$and": [{"machine": {"$eq": "X"}}, {"type_doc": {"$eq": "Y"}}]}
    Qdrant   : Filter(must=[FieldCondition(key="machine", match=MatchValue(value="GEMINI"))])

    Supporte : $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte, $and, $or
    """
    if not filtre:
        return None
    from qdrant_client.models import (
        FieldCondition, Filter, MatchAny, MatchExcept, MatchValue, Range,
    )

    def _condition(key: str, op: dict):
        if "$eq" in op:
            return FieldCondition(key=key, match=MatchValue(value=op["$eq"]))
        if "$ne" in op:
            return FieldCondition(key=key, match=MatchExcept(except_=[op["$ne"]]))
        if "$in" in op:
            return FieldCondition(key=key, match=MatchAny(any=op["$in"]))
        if "$nin" in op:
            return FieldCondition(key=key, match=MatchExcept(except_=op["$nin"]))
        range_kwargs = {
            field: op[op_key]
            for op_key, field in [("$gt", "gt"), ("$gte", "gte"), ("$lt", "lt"), ("$lte", "lte")]
            if op_key in op
        }
        if range_kwargs:
            return FieldCondition(key=key, range=Range(**range_kwargs))
        return None

    def _parse(f: dict) -> Filter | None:
        if "$and" in f:
            conds = []
            for sub in f["$and"]:
                sub_f = _parse(sub)
                if sub_f:
                    conds.extend(sub_f.must or [])
            return Filter(must=conds) if conds else None
        if "$or" in f:
            conds = []
            for sub in f["$or"]:
                sub_f = _parse(sub)
                if sub_f:
                    conds.extend(sub_f.must or sub_f.should or [])
            return Filter(should=conds) if conds else None
        conds = [
            c
            for key, op in f.items()
            if isinstance(op, dict)
            for c in [_condition(key, op)]
            if c
        ]
        return Filter(must=conds) if conds else None

    return _parse(filtre)


class QdrantCollectionStore:
    """
    Wrapper Qdrant duck-typant l'interface Chroma utilisée dans search.py et document_manager.py.

    Fournit :
      similarity_search_with_score(query, k, filter) → list[(Document, float)]
      add_texts(texts, metadatas, ids)               → list[str]
      delete(ids)                                    → None
      count()                                        → int  (remplace _collection.count())
    """

    def __init__(self, client, collection_name: str, embeddings, sparse_embeddings=None):
        from langchain_qdrant import QdrantVectorStore, RetrievalMode
        self._client = client
        self._collection_name = collection_name
        self._sparse_embeddings = sparse_embeddings

        if sparse_embeddings and EMBED_SPARSE:
            self._store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
            )
            logger.info(f"QdrantCollectionStore '{collection_name}' : mode HYBRID (dense+sparse)")
        else:
            self._store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
            )
            logger.info(f"QdrantCollectionStore '{collection_name}' : mode DENSE")

    def similarity_search_with_score(self, query: str, k: int = 6, filter: dict | None = None):
        """Recherche vectorielle + score. Convertit le filtre ChromaDB → Qdrant Filter."""
        qdrant_filter = _convertir_filtre(filter) if filter else None
        return self._store.similarity_search_with_score(query, k=k, filter=qdrant_filter)

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Indexe des textes dans la collection Qdrant.

        Si EMBED_SPARSE=true : upsert manuel avec BM25 field weighting.
        Le texte sparse contient le titre de section répété ×3 (boost TF naturel)
        tandis que le texte dense reste propre (qualité d'embedding préservée).
        """
        if self._sparse_embeddings and EMBED_SPARSE:
            return self._add_texts_title_boost(texts, metadatas or [], ids or [])
        return self._store.add_texts(texts=texts, metadatas=metadatas or [], ids=ids)

    def _add_texts_title_boost(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> list[str]:
        """
        Upsert Qdrant avec dense propre + sparse title-boosted (BM25F Option B).

        Texte dense  : texte propre → embedding sémantique de qualité
        Texte sparse : "TITRE TITRE TITRE\\n<texte>" → TF×3 du titre → boost SPLADE naturel
        Les chunks dont la section matche la query remontent plus haut dans le hybrid search.
        """
        import json as _json
        from qdrant_client.models import PointStruct
        from qdrant_client.models import SparseVector as QSparseVector
        from core.embeddings import get_embeddings

        # Dense embeddings sur texte propre
        dense_vecs = get_embeddings().embed_documents(texts)

        # Textes boostés pour le sparse : préfixer avec titre de section ×3
        nb_boosted = 0
        texts_sparse = []
        for text, meta in zip(texts, metadatas):
            parents_raw = (meta or {}).get("hierarchy_parents", "[]")
            try:
                parents = _json.loads(parents_raw) if isinstance(parents_raw, str) else parents_raw
            except Exception:
                parents = []
            titre = parents[-1] if parents else ""
            if titre:
                texts_sparse.append(f"{titre} {titre} {titre}\n{text}")
                nb_boosted += 1
            else:
                texts_sparse.append(text)

        sparse_vecs = self._sparse_embeddings.embed_documents(texts_sparse)

        # Construction des points Qdrant avec les deux vecteurs
        points = []
        for i, point_id in enumerate(ids):
            sv = sparse_vecs[i]
            points.append(PointStruct(
                id=point_id,
                vector={
                    "": dense_vecs[i],
                    "langchain-sparse": QSparseVector(
                        indices=sv.indices,
                        values=sv.values,
                    ),
                },
                payload={
                    "page_content": texts[i],
                    "metadata": metadatas[i] if metadatas else {},
                },
            ))

        # Upsert par batch de 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection_name,
                points=points[i:i + batch_size],
                wait=True,
            )

        logger.info(
            f"BM25 field boost : {nb_boosted}/{len(texts)} chunks "
            f"avec titre de section boosté ×3 dans le sparse"
        )
        return ids

    def delete(self, ids: list[str]) -> None:
        """Supprime des points par leur ID."""
        self._store.delete(ids=ids)

    def count(self) -> int:
        """Nombre de vecteurs dans la collection (remplace db._collection.count())."""
        try:
            result = self._client.count(self._collection_name)
            return result.count
        except Exception:
            return 0


class QdrantCollectionManager:
    """
    Gère les collections Qdrant — même interface que CollectionManager (ChromaDB).

    Méthodes : collection_existe(), creer_collection(), get_collection(),
               lister_collections(), supprimer_collection()

    La collection Qdrant est créée automatiquement par langchain-qdrant lors du
    premier add_texts() (dimensions inférées depuis le premier embedding).
    creer_collection() retourne simplement un QdrantCollectionStore prêt à l'emploi.
    """

    def __init__(self, host: str | None = None, port: int | None = None):
        from qdrant_client import QdrantClient
        self.host = host or QDRANT_HOST
        self.port = port or QDRANT_PORT
        self._client = QdrantClient(host=self.host, port=self.port)

    def _sparse_embeddings(self):
        if not EMBED_SPARSE:
            return None
        try:
            from core.embeddings import get_sparse_embeddings
            return get_sparse_embeddings()
        except Exception as e:
            logger.warning(f"Sparse embeddings indisponibles ({e}) — mode DENSE")
            return None

    def _create_payload_indexes(self, nom: str) -> None:
        """
        Crée les payload indexes nécessaires pour les filtres metadata.
        Doit être appelé UNE FOIS à la création de la collection (avant tout upload)
        pour que Qdrant construise des liens HNSW filter-aware.
        """
        from qdrant_client.models import PayloadSchemaType
        fields = [
            ("source",              PayloadSchemaType.KEYWORD),
            ("chunk_idx",           PayloadSchemaType.INTEGER),
            ("chunk_idx_in_doc",    PayloadSchemaType.INTEGER),
            ("total_chunks_in_doc", PayloadSchemaType.INTEGER),
            ("is_first_chunk",      PayloadSchemaType.BOOL),
            ("is_last_chunk",       PayloadSchemaType.BOOL),
            ("content_type",        PayloadSchemaType.KEYWORD),
            ("chunk_level",         PayloadSchemaType.KEYWORD),
            ("parent_id",           PayloadSchemaType.KEYWORD),
            ("page",                PayloadSchemaType.INTEGER),
        ]
        for field_name, field_type in fields:
            try:
                self._client.create_payload_index(
                    collection_name=nom,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                logger.warning(f"Payload index '{field_name}' : {e}")
        logger.info(f"Payload indexes créés pour '{nom}' ({len(fields)} champs)")

    def collection_existe(self, nom: str) -> bool:
        try:
            self._client.get_collection(nom)
            return True
        except Exception:
            return False

    def creer_collection(self, nom: str) -> QdrantCollectionStore:
        """
        Crée la collection Qdrant physiquement si elle n'existe pas, puis retourne
        un QdrantCollectionStore. langchain-qdrant valide la collection dans __init__
        donc elle doit exister avant l'instanciation.
        """
        from core.embeddings import get_embeddings
        from qdrant_client.models import Distance, VectorParams

        if not self.collection_existe(nom):
            emb = get_embeddings()
            # Inférer la dimension via un embedding test (appelé une seule fois à la création)
            dim = len(emb.embed_query("test"))
            vectors_config = {"": VectorParams(size=dim, distance=Distance.COSINE)}
            kwargs = {}
            if EMBED_SPARSE:
                from qdrant_client.models import SparseVectorParams
                kwargs["sparse_vectors_config"] = {"langchain-sparse": SparseVectorParams()}
            self._client.create_collection(
                collection_name=nom,
                vectors_config=vectors_config,
                **kwargs,
            )
            logger.info(f"Collection Qdrant '{nom}' créée ({dim}-dim, sparse={EMBED_SPARSE})")
            # Payload indexes — créés avant le premier upload pour que Qdrant puisse
            # construire des liens HNSW filter-aware (filtrage O(1) au lieu de post-filter).
            self._create_payload_indexes(nom)

        return QdrantCollectionStore(
            client=self._client,
            collection_name=nom,
            embeddings=get_embeddings(),
            sparse_embeddings=self._sparse_embeddings(),
        )

    def get_collection(self, nom: str) -> QdrantCollectionStore:
        """Retourne un QdrantCollectionStore pour une collection existante."""
        if not self.collection_existe(nom):
            raise ValueError(f"Collection '{nom}' introuvable.")
        from core.embeddings import get_embeddings
        return QdrantCollectionStore(
            client=self._client,
            collection_name=nom,
            embeddings=get_embeddings(),
            sparse_embeddings=self._sparse_embeddings(),
        )

    def lister_collections(self) -> list[str]:
        """Liste toutes les collections disponibles."""
        return sorted(c.name for c in self._client.get_collections().collections)

    def supprimer_collection(self, nom: str) -> None:
        """Supprime une collection Qdrant."""
        self._client.delete_collection(nom)
