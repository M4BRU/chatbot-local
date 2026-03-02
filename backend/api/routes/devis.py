"""Devis & catalog API routes."""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.api.dependencies import get_catalog_adapter, get_devis_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["devis"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ── Request models ─────────────────────────────────────────────────────────────

class DevisChatRequest(BaseModel):
    message: str
    collection: str = ""
    conversation_id: str
    history: list[dict] = []


class GenerateDevisRequest(BaseModel):
    conversation_id: str


class LockAffaireRequest(BaseModel):
    nom_affaire: str


class AddElementRequest(BaseModel):
    elements: str
    nom_poste: str = ""
    nom_affaire: str = ""
    fournisseur: str = ""
    fourniture: str = ""
    ensemble: str = ""
    num_affaire: str = ""
    num_poste: str = ""


class SearchScopeRequest(BaseModel):
    search_all: bool


# ── Catalog management ─────────────────────────────────────────────────────────

@router.get("/catalog/status")
async def catalog_status(catalog=Depends(get_catalog_adapter)):
    """Return catalog metadata (loaded, row count, columns)."""
    return catalog.status


@router.post("/catalog/upload")
async def upload_catalog(
    file: UploadFile = File(...),
    catalog=Depends(get_catalog_adapter),
):
    """Upload an Excel catalog file and load it into SQLite."""
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400, detail="Format attendu : .xlsx ou .xls"
        )

    dest = Path("/app/documents/catalogue.xlsx")
    content = await file.read()
    dest.write_bytes(content)

    result = await asyncio.to_thread(catalog.load_from_excel, dest)
    return {"status": "ok", **result}


@router.get("/catalog/elements")
async def get_catalog_elements(
    nom_poste: str,
    nom_affaire: str | None = None,
    catalog=Depends(get_catalog_adapter),
):
    """Return all catalog sub-rows for a given nom_poste, optionally filtered by affaire."""
    return await asyncio.to_thread(
        catalog.get_elements_for_poste, nom_poste, nom_affaire
    )


@router.post("/catalog/reload")
async def reload_catalog(catalog=Depends(get_catalog_adapter)):
    """Reload the catalog from disk (useful after manual file replacement)."""
    catalog_path = Path("/app/documents/catalogue.xlsx")
    if not catalog_path.exists():
        raise HTTPException(status_code=404, detail="Fichier catalogue introuvable sur le disque.")

    result = await asyncio.to_thread(catalog.load_from_excel, catalog_path)
    return {"status": "ok", **result}


# ── Devis chat ─────────────────────────────────────────────────────────────────

@router.post("/devis/chat")
async def devis_chat(
    req: DevisChatRequest,
    devis_service=Depends(get_devis_service),
):
    """SSE stream for a devis chat turn (tool-calling loop + final generation)."""

    async def _generator():
        async for event in devis_service.chat_stream(
            message=req.message,
            collection=req.collection,
            conversation_id=req.conversation_id,
            history=req.history,
        ):
            yield {"data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(_generator(), headers=_SSE_HEADERS)


@router.post("/devis/generate")
async def generate_devis(
    req: GenerateDevisRequest,
    devis_service=Depends(get_devis_service),
):
    """SSE stream that generates the final devis markdown table from the panier."""

    async def _generator():
        async for event in devis_service.generate_devis(req.conversation_id):
            yield {"data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(_generator(), headers=_SSE_HEADERS)


# ── Panier management ──────────────────────────────────────────────────────────

@router.get("/devis/{conversation_id}/panier")
async def get_panier(
    conversation_id: str,
    catalog=Depends(get_catalog_adapter),
):
    """Return all panier items for a conversation."""
    return await asyncio.to_thread(catalog.get_panier, conversation_id)


@router.post("/devis/{conversation_id}/lock-affaire")
async def lock_affaire(
    conversation_id: str,
    req: LockAffaireRequest,
    catalog=Depends(get_catalog_adapter),
):
    """Lock an affaire for a conversation (called when user clicks a choice card)."""
    await asyncio.to_thread(catalog.set_affaire_lock, conversation_id, req.nom_affaire)
    return {"status": "locked", "nom_affaire": req.nom_affaire}


@router.post("/devis/{conversation_id}/search-scope")
async def set_search_scope(
    conversation_id: str,
    req: SearchScopeRequest,
    catalog=Depends(get_catalog_adapter),
):
    """Set whether the next search_catalog call should return all affaires or only the locked one."""
    await asyncio.to_thread(catalog.set_search_scope, conversation_id, req.search_all)
    return {"status": "ok", "search_all": req.search_all}


@router.delete("/devis/{conversation_id}/panier")
async def clear_panier(
    conversation_id: str,
    catalog=Depends(get_catalog_adapter),
):
    """Remove all items from a conversation's panier."""
    await asyncio.to_thread(catalog.clear_panier, conversation_id)
    return {"status": "cleared"}


@router.get("/devis/{conversation_id}/export")
async def export_devis_excel(
    conversation_id: str,
    catalog=Depends(get_catalog_adapter),
):
    """Generate and download a filled Excel devis from the panier."""
    from backend.domain.services.excel_generator import generate_excel_devis

    panier = await asyncio.to_thread(catalog.get_panier, conversation_id)
    if not panier:
        raise HTTPException(status_code=400, detail="Le panier est vide — ajoutez des produits avant de générer le devis.")

    try:
        xlsx_bytes = await asyncio.to_thread(generate_excel_devis, panier, catalog)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    filename = f"devis_{conversation_id[:8]}.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/devis/{conversation_id}/panier/element")
async def add_element_to_panier(
    conversation_id: str,
    req: AddElementRequest,
    catalog=Depends(get_catalog_adapter),
):
    """Add an element (col H sub-component) directly to the panier without LLM."""
    result = await asyncio.to_thread(
        catalog.add_element_to_panier, conversation_id, req.model_dump()
    )
    if result.get("errors"):
        raise HTTPException(status_code=400, detail=result["errors"][0].get("error"))
    return result


@router.delete("/devis/{conversation_id}/panier/{item_id}")
async def remove_panier_item(
    conversation_id: str,
    item_id: str,
    catalog=Depends(get_catalog_adapter),
):
    """Remove a specific item from the panier."""
    await asyncio.to_thread(catalog.remove_from_panier, conversation_id, item_id)
    return {"status": "removed"}
