"""Collections management API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import get_collection_manager

router = APIRouter(prefix="/api/collections", tags=["collections"])


class CollectionCreate(BaseModel):
    """Request model for creating a collection."""

    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")


class CollectionInfo(BaseModel):
    """Collection information."""

    name: str
    document_count: int


class CollectionListResponse(BaseModel):
    """Response for listing collections."""

    collections: list[str]


@router.get("", response_model=CollectionListResponse)
async def list_collections() -> CollectionListResponse:
    """List all available collections."""
    cm = get_collection_manager()
    return CollectionListResponse(collections=cm.lister_collections())


@router.post("", response_model=CollectionInfo, status_code=201)
async def create_collection(request: CollectionCreate) -> CollectionInfo:
    """Create a new collection."""
    cm = get_collection_manager()
    if cm.collection_existe(request.name):
        raise HTTPException(status_code=409, detail=f"Collection '{request.name}' already exists")

    cm.creer_collection(request.name)
    return CollectionInfo(name=request.name, document_count=0)


@router.get("/{name}", response_model=CollectionInfo)
async def get_collection(name: str) -> CollectionInfo:
    """Get collection information."""
    cm = get_collection_manager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    db = cm.get_collection(name)
    # Get document count from ChromaDB
    try:
        count = db._collection.count()
    except Exception:
        count = 0

    return CollectionInfo(name=name, document_count=count)


@router.get("/{name}/sources")
async def list_sources(name: str) -> dict:
    """Liste les noms de fichiers uniques indexés dans une collection."""
    cm = get_collection_manager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    db = cm.get_collection(name)
    try:
        sources: set[str] = set()
        if hasattr(db, "_client"):
            # Qdrant : scroll complet pour collecter les sources uniques
            next_offset = None
            while True:
                records, next_offset = db._client.scroll(
                    collection_name=name,
                    limit=500,
                    offset=next_offset,
                    with_payload=["metadata"],
                    with_vectors=False,
                )
                for record in records:
                    meta = (record.payload or {}).get("metadata", {}) or {}
                    src = meta.get("source")
                    if src:
                        sources.add(src)
                if not next_offset or not records:
                    break
        else:
            result = db._collection.get(include=["metadatas"])
            for meta in (result.get("metadatas") or []):
                src = (meta or {}).get("source")
                if src:
                    sources.add(src)

        return {"sources": sorted(sources)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}/chunks")
async def list_chunks(name: str, offset: int = 0, limit: int = 50, source: str = "") -> dict:
    """Liste les chunks d'une collection avec pagination. Filtre optionnel par source."""
    cm = get_collection_manager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    db = cm.get_collection(name)
    try:
        import json as _json
        chunks = []

        if hasattr(db, "_client"):
            # ── Qdrant ──────────────────────────────────────────────────────
            qdrant_filter = None
            if source:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                qdrant_filter = Filter(must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value=source))
                ])

            if source:
                # Filtre par fichier : charge tous les chunks du fichier d'un coup (< 1000)
                records, _ = db._client.scroll(
                    collection_name=name,
                    limit=2000,
                    offset=None,
                    scroll_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=False,
                )
                total = len(records)
            else:
                total = db.count()
                records, _ = db._client.scroll(
                    collection_name=name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

            for record in records:
                payload = record.payload or {}
                meta = payload.get("metadata", {}) or {}
                text = payload.get("page_content", "")
                sections_raw = meta.get("hierarchy_parents", "[]")
                try:
                    sections = _json.loads(sections_raw) if isinstance(sections_raw, str) else sections_raw
                except Exception:
                    sections = []
                chunks.append({
                    "source": meta.get("source", "?"),
                    "page": meta.get("page", "?"),
                    "chunk_idx": meta.get("chunk_idx"),
                    "machine": meta.get("machine"),
                    "sections": sections,
                    "content": text,
                    "content_preview": text[:300] if text else "",
                    "parent_text": meta.get("parent_text"),
                })
        else:
            # ── ChromaDB ────────────────────────────────────────────────────
            total = db._collection.count()
            result = db._collection.get(
                include=["documents", "metadatas"],
                limit=limit,
                offset=offset,
                where={"source": source} if source else None,
            )
            texts = result.get("documents") or []
            metas = result.get("metadatas") or []
            for text, meta in zip(texts, metas):
                meta = meta or {}
                sections_raw = meta.get("hierarchy_parents", "[]")
                try:
                    sections = _json.loads(sections_raw) if isinstance(sections_raw, str) else sections_raw
                except Exception:
                    sections = []
                chunks.append({
                    "source": meta.get("source", "?"),
                    "page": meta.get("page", "?"),
                    "chunk_idx": meta.get("chunk_idx"),
                    "machine": meta.get("machine"),
                    "sections": sections,
                    "content": text,
                    "content_preview": text[:300] if text else "",
                    "parent_text": meta.get("parent_text"),
                })

        chunks.sort(key=lambda c: (c["source"], int(c["page"]) if str(c["page"]).isdigit() else 0, c["chunk_idx"] or 0))
        return {"total": total, "offset": offset, "limit": limit, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/version-status")
async def version_status() -> list[dict]:
    """
    Vérifie si les collections ont des chunks périmés (pipeline d'indexation modifié).
    Lecture des metadata.json uniquement — pas de requête vector DB.
    """
    from backend.api.dependencies import get_document_manager
    dm = get_document_manager()
    return dm.verifier_versions_toutes_collections()


@router.delete("/{name}", status_code=204)
async def delete_collection(name: str) -> None:
    """Delete a collection."""
    cm = get_collection_manager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    cm.supprimer_collection(name)
