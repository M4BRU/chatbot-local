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


@router.get("/{name}/chunks")
async def list_chunks(name: str, offset: int = 0, limit: int = 50) -> dict:
    """Liste les chunks d'une collection avec pagination."""
    cm = get_collection_manager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    db = cm.get_collection(name)
    try:
        import json as _json
        total = db._collection.count()
        result = db._collection.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )
        texts = result.get("documents") or []
        metas = result.get("metadatas") or []

        chunks = []
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
            })

        chunks.sort(key=lambda c: (c["source"], c["chunk_idx"] or 0))
        return {"total": total, "offset": offset, "limit": limit, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}", status_code=204)
async def delete_collection(name: str) -> None:
    """Delete a collection."""
    cm = get_collection_manager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    cm.supprimer_collection(name)
