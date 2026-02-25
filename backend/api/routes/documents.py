"""Documents management API routes."""

import asyncio
import shutil
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.api.dependencies import get_collection_manager, get_settings

router = APIRouter(prefix="/api/collections/{collection_name}/documents", tags=["documents"])

settings = get_settings()


class DocumentInfo(BaseModel):
    """Document information."""

    nom: str
    date: str
    nb_chunks: int
    nb_pages: int


class DocumentListResponse(BaseModel):
    """Response for listing documents."""

    documents: list[DocumentInfo]


class IndexResult(BaseModel):
    """Result of document indexation."""

    status: str
    chunks: int
    message: str
    warnings: list[str] = []


@router.get("", response_model=DocumentListResponse)
async def list_documents(collection_name: str) -> DocumentListResponse:
    """List all documents in a collection."""
    from core.document_manager import DocumentManager

    cm = get_collection_manager()
    if not cm.collection_existe(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    dm = DocumentManager(cm)
    docs = dm.lister_documents(collection_name)
    return DocumentListResponse(documents=[DocumentInfo(**d) for d in docs])


@router.post("", response_model=IndexResult, status_code=201)
async def upload_document(
    collection_name: str,
    file: UploadFile = File(...),
    force: bool = Query(False, description="Force re-indexation even if document exists"),
) -> IndexResult:
    """
    Upload and index a document in a collection.

    Supported formats: PDF, TXT, MD, DOCX
    """
    from core.document_manager import DocumentManager

    cm = get_collection_manager()

    # Create collection if it doesn't exist
    if not cm.collection_existe(collection_name):
        cm.creer_collection(collection_name)

    # Sanitize filename pour éviter le path traversal
    # PurePosixPath.name extrait uniquement le nom de fichier, sans répertoire
    raw_name = file.filename or "document"
    safe_name = PurePosixPath(raw_name).name
    if not safe_name or safe_name in (".", ".."):
        safe_name = "document"

    suffix = Path(safe_name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    final_path = tmp_path.parent / safe_name
    try:
        tmp_path.rename(final_path)

        dm = DocumentManager(cm)
        result = await asyncio.to_thread(dm.ajouter_document, collection_name, final_path, force)
        return IndexResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        for p in [tmp_path, final_path]:
            if p.exists():
                p.unlink()


@router.delete("/{document_name}", status_code=204)
async def delete_document(collection_name: str, document_name: str) -> None:
    """Delete a document from a collection."""
    from core.document_manager import DocumentManager

    cm = get_collection_manager()
    if not cm.collection_existe(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    dm = DocumentManager(cm)
    if not dm.supprimer_document(collection_name, document_name):
        raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
