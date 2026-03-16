"""Devis & catalog API routes."""

import asyncio
import json
import logging
import time
from pathlib import Path

import requests as _requests
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse

from backend.api.dependencies import get_catalog_adapter, get_devis_service, get_excel_collection_adapter, get_settings

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
    catalog_method: str = "bm25"  # "bm25" | "sql"


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


class AddPosteRequest(BaseModel):
    nom_poste: str
    nom_affaire: str = ""
    num_poste: str = ""
    quantite: int = 1


class SearchScopeRequest(BaseModel):
    search_all: bool


class DevisSettingsRequest(BaseModel):
    coefficient: float = 0.0
    coef_final:  float = 0.0


class UpdatePanierItemRequest(BaseModel):
    nbre_jours_etude:   int  | None = None
    nbre_jours_atelier: int  | None = None
    nbre_jours_client:  int  | None = None
    is_option:          bool | None = None

    @field_validator("nbre_jours_etude", "nbre_jours_atelier", "nbre_jours_client")
    @classmethod
    def validate_non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("Must be >= 0")
        return v


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


# ── Catalog challenge ──────────────────────────────────────────────────────────

_CATALOG_CHALLENGE_COLLECTION = "_catalog_challenge"
_CATALOG_PATH = Path("/app/documents/catalogue.xlsx")


class CatalogChallengeRequest(BaseModel):
    question: str
    with_synthesis: bool = False


def _synthesize(question: str, context: str) -> str:
    """Appel Ollama synchrone pour synthétiser une réponse."""
    prompt = (
        "Tu réponds en français de façon concise (2-4 phrases max).\n\n"
        f"Question : {question}\n\n"
        f"Données disponibles :\n{context}\n\n"
        "Réponds directement. Si les données sont vides, dis-le clairement."
    )
    try:
        s = get_settings()
        resp = _requests.post(
            f"{s.ollama_url}/api/generate",
            json={"model": s.ollama_model, "prompt": prompt, "stream": False,
                  "think": False, "options": {"temperature": 0.1, "num_predict": 300}},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"(erreur synthèse : {e})"


@router.post("/catalog/challenge")
async def catalog_challenge(
    req: CatalogChallengeRequest,
    catalog=Depends(get_catalog_adapter),
    excel_adapter=Depends(get_excel_collection_adapter),
) -> dict:
    """
    Compare BM25 (méthode catalogue actuelle) vs NL2SQL (nouvelle méthode SQL)
    sur le fichier catalogue. Retourne résultats + temps de réponse.
    """
    question = req.question

    # ── Pipeline BM25 ─────────────────────────────────────────────────────────
    t0 = time.time()
    bm25_rows = await asyncio.to_thread(catalog.search, question, 10)
    bm25_time = round(time.time() - t0, 3)

    # ── Pipeline SQL — auto-indexe le catalogue au premier appel ──────────────
    sql_error = None
    if not excel_adapter.has_excel_data(_CATALOG_CHALLENGE_COLLECTION):
        if not _CATALOG_PATH.exists():
            sql_error = "Fichier catalogue introuvable (/app/documents/catalogue.xlsx)"
        else:
            logger.info("Catalog challenge : indexation SQL du catalogue…")
            await asyncio.to_thread(excel_adapter.ingest, _CATALOG_PATH, _CATALOG_CHALLENGE_COLLECTION)

    if sql_error:
        sql_result = {"method": "error", "sql": "", "results": [], "error": sql_error}
        sql_time = 0.0
    else:
        t0 = time.time()
        sql_result = await asyncio.to_thread(
            excel_adapter.nl2sql_query, question, _CATALOG_CHALLENGE_COLLECTION
        )
        sql_time = round(time.time() - t0, 3)

    bm25_section: dict = {
        "method": "bm25",
        "results": bm25_rows,
        "count": len(bm25_rows),
        "time_s": bm25_time,
        "llm_answer": None,
    }
    sql_section: dict = {
        "method": sql_result.get("method", "unknown"),
        "query_generated": sql_result.get("sql", ""),
        "results": sql_result.get("results", []),
        "count": len(sql_result.get("results", [])),
        "time_s": sql_time,
        "error": sql_result.get("error"),
        "llm_answer": None,
    }

    if req.with_synthesis:
        bm25_ctx = json.dumps(bm25_rows[:5], ensure_ascii=False, default=str)
        bm25_section["llm_answer"] = await asyncio.to_thread(
            _synthesize, question, f"Résultats BM25 :\n{bm25_ctx}"
        )
        sql_ctx = json.dumps(sql_result.get("results", [])[:10], ensure_ascii=False, default=str)
        if not sql_result.get("results"):
            sql_ctx = sql_result.get("error") or "Aucun résultat"
        sql_section["llm_answer"] = await asyncio.to_thread(
            _synthesize, question, f"Résultats SQL :\n{sql_ctx}"
        )

    return {"question": question, "bm25": bm25_section, "sql": sql_section}


# ── RFQ Planner debug ──────────────────────────────────────────────────────────

class RfqDebugRequest(BaseModel):
    message: str
    collection: str


@router.post("/devis/rfq-debug")
async def rfq_planner_debug(
    req: RfqDebugRequest,
    devis_service=Depends(get_devis_service),
) -> dict:
    """
    Exécute le RFQPlanner en mode debug et retourne la structure complète :
    - Phase 1 : dimensions extraites
    - Phase 2 : chunks RAG récupérés + composants extraits par dimension
    - Phase 3 : gaps détectés + recherches complémentaires
    - Phase 4 : contexte synthétisé final
    Permet de diagnostiquer ce que le planner trouve (ou rate) à chaque étape.
    """
    return await devis_service.run_rfq_planner_debug(req.message, req.collection)


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
            catalog_method=req.catalog_method,
        ):
            yield {"data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(_generator(), headers=_SSE_HEADERS, ping=15)


@router.post("/devis/generate")
async def generate_devis(
    req: GenerateDevisRequest,
    devis_service=Depends(get_devis_service),
):
    """SSE stream that generates the final devis markdown table from the panier."""

    async def _generator():
        async for event in devis_service.generate_devis(req.conversation_id):
            yield {"data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(_generator(), headers=_SSE_HEADERS, ping=15)


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


@router.get("/devis/{conversation_id}/settings")
async def get_devis_settings(
    conversation_id: str,
    catalog=Depends(get_catalog_adapter),
):
    """Return devis settings (coefficient, coef_final) for a conversation."""
    return await asyncio.to_thread(catalog.get_devis_settings, conversation_id)


@router.patch("/devis/{conversation_id}/settings")
async def patch_devis_settings(
    conversation_id: str,
    req: DevisSettingsRequest,
    catalog=Depends(get_catalog_adapter),
):
    """Update devis settings (coefficient, coef_final) for a conversation."""
    await asyncio.to_thread(catalog.set_devis_settings, conversation_id, req.coefficient, req.coef_final)
    return {"status": "ok", "coefficient": req.coefficient, "coef_final": req.coef_final}


@router.patch("/devis/{conversation_id}/panier/{item_id}")
async def patch_panier_item(
    conversation_id: str,
    item_id: str,
    req: UpdatePanierItemRequest,
    catalog=Depends(get_catalog_adapter),
):
    """Partial update of a panier item (MdO fields + is_option)."""
    fields = req.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        updated = await asyncio.to_thread(catalog.update_panier_item, conversation_id, item_id, fields)
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    return updated


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

    settings = await asyncio.to_thread(catalog.get_devis_settings, conversation_id)
    main_items   = [p for p in panier if not p.get("is_option")]
    option_items = [p for p in panier if p.get("is_option")]

    # Inject global coef_final into every item dict
    for item in main_items + option_items:
        item["coef_final"] = settings.get("coef_final", 0.0)

    try:
        xlsx_bytes = await asyncio.to_thread(
            generate_excel_devis,
            main_items,
            catalog,
            coefficient=settings.get("coefficient", 0.0),
            options=option_items if option_items else None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    filename = f"devis_{conversation_id[:8]}.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/devis/{conversation_id}/panier/poste")
async def add_poste_to_panier_direct(
    conversation_id: str,
    req: AddPosteRequest,
    catalog=Depends(get_catalog_adapter),
):
    """Add a poste directly to the panier without LLM (called from UI choice cards)."""
    result = await asyncio.to_thread(
        catalog.add_to_panier,
        conversation_id,
        [req.model_dump()],
    )
    if result.get("errors"):
        first_error = result["errors"][0]
        raise HTTPException(status_code=400, detail=first_error.get("error", "Erreur inconnue"))
    # Mark matching task as done so the LLM's LISTE DE TÂCHES is updated
    for item in result.get("added", []):
        nom = item.get("nom_poste", "") if isinstance(item, dict) else str(item)
        if nom:
            await asyncio.to_thread(catalog.complete_task_item, conversation_id, nom)
    # Include remaining tasks so the frontend can build an explicit next-search instruction
    all_tasks = await asyncio.to_thread(catalog.get_task_list, conversation_id)
    remaining = [t for t in all_tasks if not t.get("done")]
    return {**result, "remaining_tasks": remaining}


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
