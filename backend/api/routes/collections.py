"""Collections management API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
    from core.collection_manager import CollectionManager

    cm = CollectionManager()
    return CollectionListResponse(collections=cm.lister_collections())


@router.post("", response_model=CollectionInfo, status_code=201)
async def create_collection(request: CollectionCreate) -> CollectionInfo:
    """Create a new collection."""
    from core.collection_manager import CollectionManager

    cm = CollectionManager()
    if cm.collection_existe(request.name):
        raise HTTPException(status_code=409, detail=f"Collection '{request.name}' already exists")

    cm.creer_collection(request.name)
    return CollectionInfo(name=request.name, document_count=0)


@router.get("/{name}", response_model=CollectionInfo)
async def get_collection(name: str) -> CollectionInfo:
    """Get collection information."""
    from core.collection_manager import CollectionManager

    cm = CollectionManager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    db = cm.get_collection(name)
    # Get document count from ChromaDB
    try:
        count = db._collection.count()
    except Exception:
        count = 0

    return CollectionInfo(name=name, document_count=count)


@router.delete("/{name}", status_code=204)
async def delete_collection(name: str) -> None:
    """Delete a collection."""
    from core.collection_manager import CollectionManager

    cm = CollectionManager()
    if not cm.collection_existe(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    cm.supprimer_collection(name)
