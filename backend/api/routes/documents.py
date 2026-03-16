"""Documents management API routes."""

import asyncio
import json
import logging
import shutil
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

import requests as _requests
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.api.dependencies import get_collection_manager, get_excel_collection_adapter, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections/{collection_name}/documents", tags=["documents"])

settings = get_settings()

# Extensions Excel reconnues
_EXCEL_SUFFIXES = {".xlsx", ".xls"}


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
    excel_sql: dict | None = None  # Présent si le document est un Excel


class ExcelTableInfo(BaseModel):
    """Info sur une table Excel indexée dans SQLite."""

    id: str
    filename: str
    sheet_name: str
    table_name: str
    columns: list[dict]
    sample_rows: list[dict]
    row_count: int
    uploaded_at: str


class ExcelListResponse(BaseModel):
    tables: list[ExcelTableInfo]


class ExcelCompareRequest(BaseModel):
    question: str
    with_synthesis: bool = False
    filename: str | None = None  # Si renseigné, restreint SQL + BM25 à ce fichier uniquement


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
    excel_adapter=Depends(get_excel_collection_adapter),
) -> IndexResult:
    """
    Upload and index a document in a collection.

    Supported formats: PDF, TXT, MD, DOCX, XLSX, XLS

    Pour les fichiers Excel (.xlsx / .xls) :
      - Pipeline vecteur (existant) : parsers.py → Qdrant/Chroma
      - Pipeline SQL (nouveau)      : SQLite dynamique + NL2SQL (déterministe)
    Les deux pipelines sont alimentés automatiquement, sans action supplémentaire.
    """
    from core.document_manager import DocumentManager

    cm = get_collection_manager()

    # Create collection if it doesn't exist
    if not cm.collection_existe(collection_name):
        cm.creer_collection(collection_name)

    # Sanitize filename pour éviter le path traversal
    raw_name = file.filename or "document"
    safe_name = PurePosixPath(raw_name).name
    if not safe_name or safe_name in (".", ".."):
        safe_name = "document"

    suffix = Path(safe_name).suffix.lower()
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    final_path = tmp_path.parent / safe_name
    excel_sql_result: dict | None = None

    try:
        tmp_path.rename(final_path)

        if suffix in _EXCEL_SUFFIXES:
            # ── Excel : SQL uniquement (pas de vecteur) ───────────────────────
            # Les données tabulaires sont mieux servies par NL2SQL que par RAG vecteur.
            try:
                excel_sql_result = await asyncio.to_thread(
                    excel_adapter.ingest, final_path, collection_name
                )
            except Exception as e:
                logger.warning("Excel SQL ingest échoué pour %s : %s", safe_name, e)
                excel_sql_result = {"error": str(e)}
            return IndexResult(
                status="ok",
                chunks=0,
                message="Indexé SQL (NL2SQL) — pipeline vecteur désactivé pour les Excel",
                excel_sql=excel_sql_result,
            )

        # ── Pipeline vecteur (PDF, DOCX, TXT, MD, CSV…) ──────────────────────
        dm = DocumentManager(cm)
        result = await asyncio.to_thread(dm.ajouter_document, collection_name, final_path, force)
        return IndexResult(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [tmp_path, final_path]:
            if p.exists():
                p.unlink()


@router.delete("/{document_name}", status_code=204)
async def delete_document(
    collection_name: str,
    document_name: str,
    excel_adapter=Depends(get_excel_collection_adapter),
) -> None:
    """Delete a document from a collection (vecteur + SQL si Excel)."""
    from core.document_manager import DocumentManager

    cm = get_collection_manager()
    if not cm.collection_existe(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    suffix = Path(document_name).suffix.lower()
    dm = DocumentManager(cm)
    if suffix in _EXCEL_SUFFIXES:
        # Essaie SQL d'abord, fallback vecteur (pour les Excel indexés avant la migration SQL)
        deleted = excel_adapter.delete_file(collection_name, document_name)
        if not deleted:
            deleted = dm.supprimer_document(collection_name, document_name)
    else:
        deleted = dm.supprimer_document(collection_name, document_name)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")


# ── Endpoints Excel SQL ───────────────────────────────────────────────────────

def _call_synthesis(question: str, context: str) -> str:
    """Appel Ollama synchrone pour synthétiser une réponse à partir de données brutes."""
    prompt = (
        "Tu réponds en français de façon concise (2-4 phrases max).\n\n"
        f"Question : {question}\n\n"
        f"Données disponibles :\n{context}\n\n"
        "Réponds directement sans préambule. "
        "Si les données sont vides ou nulles, dis-le clairement."
    )
    try:
        _settings = get_settings()
        resp = _requests.post(
            f"{_settings.ollama_url}/api/generate",
            json={
                "model": _settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Synthesis Ollama échoué : %s", e)
        return f"(erreur synthèse : {e})"


excel_router = APIRouter(
    prefix="/api/collections/{collection_name}/excel",
    tags=["excel-sql"],
)


@excel_router.get("", response_model=ExcelListResponse)
async def list_excel_tables(
    collection_name: str,
    excel_adapter=Depends(get_excel_collection_adapter),
) -> ExcelListResponse:
    """
    Liste les fichiers Excel indexés dans le pipeline SQL pour cette collection.
    Retourne le schéma (colonnes + types) et un échantillon de 3 lignes par table.
    """
    tables = excel_adapter.list_tables(collection_name)
    return ExcelListResponse(tables=[ExcelTableInfo(**t) for t in tables])


@excel_router.post("/compare")
async def compare_excel_retrieval(
    collection_name: str,
    req: ExcelCompareRequest,
    excel_adapter=Depends(get_excel_collection_adapter),
) -> dict:
    """
    Compare les deux pipelines de retrieval Excel côte à côte :
      - sql    : NL2SQL déterministe (SQLite)
      - vector : RAG vectoriel classique (Qdrant/Chroma)

    Utile pour évaluer lequel donne de meilleurs résultats sur les données structurées.

    Exemple :
      POST /api/collections/tarifs-2024/excel/compare
      {"question": "prix du moteur 3kW ?"}
    """
    question = req.question

    # ── Pipeline SQL ──────────────────────────────────────────────────────────
    if not excel_adapter.has_excel_data(collection_name):
        sql_result = {
            "method": "empty",
            "sql": "",
            "results": [],
            "error": "Aucun fichier Excel indexé dans le pipeline SQL pour cette collection. "
                     "Uploadez un fichier Excel via POST /documents pour l'activer.",
        }
    else:
        sql_result = await asyncio.to_thread(
            excel_adapter.nl2sql_query, question, collection_name, req.filename
        )

    # ── Pipeline Vecteur ──────────────────────────────────────────────────────
    vector_result: dict = {"method": "vector_rag", "results": [], "count": 0, "error": None}
    try:
        from core.collection_manager import CollectionManager
        from core.search import RAGEngine

        cm = get_collection_manager()
        if cm.collection_existe(collection_name):
            engine = RAGEngine(collection_name, collection_manager=cm)
            contexte, sources = await asyncio.to_thread(engine.rechercher, question)
            vector_result["results"] = sources
            vector_result["count"] = len(sources)
            vector_result["contexte_preview"] = contexte[:500] if contexte else ""
        else:
            vector_result["error"] = f"Collection '{collection_name}' non trouvée dans le vecteur store"
    except Exception as e:
        vector_result["error"] = str(e)

    sql_section = {
        "method": sql_result.get("method", "unknown"),
        "query_generated": sql_result.get("sql", ""),
        "results": sql_result.get("results", []),
        "count": len(sql_result.get("results", [])),
        "error": sql_result.get("error"),
        "llm_answer": None,
    }
    if vector_result.get("results"):
        vector_result["llm_answer"] = None
    else:
        vector_result["llm_answer"] = None

    if req.with_synthesis:
        # Synthèse SQL
        sql_ctx = json.dumps(sql_result.get("results", []), ensure_ascii=False, default=str)
        if not sql_result.get("results"):
            sql_ctx = sql_result.get("error") or "Aucun résultat"
        sql_section["llm_answer"] = await asyncio.to_thread(
            _call_synthesis, question, f"Résultats SQL :\n{sql_ctx}"
        )

        # Synthèse Vecteur
        vec_chunks = vector_result.get("results", [])
        if vec_chunks:
            vec_ctx = "\n\n".join(
                f"[score {c.get('score', '?'):.3f}] {c.get('texte', '')[:400]}"
                for c in vec_chunks
            )
        else:
            vec_ctx = vector_result.get("error") or "Aucun chunk trouvé"
        vector_result["llm_answer"] = await asyncio.to_thread(
            _call_synthesis, question, f"Extraits documents :\n{vec_ctx}"
        )

    return {
        "question": question,
        "collection": collection_name,
        "sql": sql_section,
        "vector": vector_result,
    }


@excel_router.delete("/{filename}", status_code=204)
async def delete_excel_file(
    collection_name: str,
    filename: str,
    excel_adapter=Depends(get_excel_collection_adapter),
) -> None:
    """Supprime un fichier Excel du pipeline SQL (les données vecteur restent intactes)."""
    deleted = excel_adapter.delete_file(collection_name, filename)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Fichier '{filename}' non trouvé dans le pipeline SQL de la collection '{collection_name}'",
        )
