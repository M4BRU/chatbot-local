"""
Documentation tools: search_docs, report_findings, search_collection_excel.
Also contains the chunk processing / component extraction helpers.
"""

import asyncio
import json
import logging

from ..ollama_client import OLLAMA_BASE_URL, OLLAMA_MODEL, _get_http_client
from ..prompts import SCHEMA_COMPONENTS

logger = logging.getLogger(__name__)

MAX_CHUNKS = 3
MAX_CHARS_PER_CHUNK = 6000


async def extract_components_from_chunks(chunks: list[dict], original_query: str) -> dict:
    """
    Appel LLM léger pour extraire les noms de composants depuis les chunks RAG.
    Retourne {"components": [{"nom": "...", "specs": "..."}]}.
    """
    if not chunks:
        logger.info("[extract_components] aucun chunk fourni → liste vide")
        return {"components": []}

    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\n[Extrait {i + 1} — {chunk['source']}]\n{chunk['text']}\n"

    system_prompt = (
        "Tu es un extracteur de données technique. "
        "Analyse les extraits et liste TOUS les modèles/équipements mentionnés "
        "avec leurs caractéristiques (capacité, type, dimensions…).\n"
        "Réponds UNIQUEMENT avec un tableau JSON, sans texte avant ou après :\n"
        '[{"nom": "NomModele", "specs": "capacité, type, ..."}]\n'
        "Si aucun modèle identifiable : []"
    )
    user_msg = (
        f"Recherche : « {original_query} »\n\n"
        f"Extraits :{chunks_text}\n"
        "Extrais tous les modèles et leurs specs."
    )

    total_chars = len(system_prompt) + len(user_msg)
    logger.info("[extract_components] appel LLM — %d chunk(s), ~%d chars input", len(chunks), total_chars)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "think": False,
        "format": SCHEMA_COMPONENTS,
        "options": {"temperature": 0, "num_ctx": 8192},
    }

    content = ""
    try:
        client = _get_http_client()
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "").strip()
        logger.info("[extract_components] réponse brute (%d chars): %r", len(content), content[:500])

        parsed = json.loads(content)
        if isinstance(parsed, list):
            components = parsed
        else:
            components = parsed.get("components", [])
        noms = [c.get("nom", "?") for c in components if isinstance(c, dict)]
        logger.info("[extract_components] ✓ %d composant(s): %s", len(components), noms)
        return {"components": components}

    except json.JSONDecodeError as exc:
        logger.error(
            "[extract_components] JSONDecodeError inattendu (format schema actif): %s | raw=%r",
            exc, content[:300],
        )
        return {"components": [], "error": f"json_decode: {exc}"}
    except Exception as exc:
        logger.warning("[extract_components] extraction échouée: %s", exc, exc_info=True)
        return {"components": [], "error": str(exc)}


async def process_chunks_and_extract(
    context_parts: list[str],
    sources: list[dict],
    query: str,
    log_prefix: str = "chunks",
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Shared chunk processing: split context into chunks, extract components via LLM, dedup.
    Returns (chunks_raw, deduped_components, unique_sources).
    """
    chunks_raw: list[dict] = []
    all_components: list[dict] = []

    for i in range(min(MAX_CHUNKS, len(context_parts))):
        text = context_parts[i][:MAX_CHARS_PER_CHUNK]
        source = sources[i].get("fichier", "") if i < len(sources) else ""
        chunk = {"text": text, "source": source}
        chunks_raw.append(chunk)
        logger.info(
            "[%s] chunk %d/%d (%d chars) → extraction LLM",
            log_prefix, i + 1, min(MAX_CHUNKS, len(context_parts)), len(text),
        )
        extraction = await extract_components_from_chunks([chunk], query)
        all_components.extend(extraction.get("components", []))

    seen_noms: set[str] = set()
    deduped: list[dict] = []
    for comp in all_components:
        nom = comp.get("nom", "")
        if nom and nom not in seen_noms:
            seen_noms.add(nom)
            deduped.append(comp)
        elif not nom:
            deduped.append(comp)

    unique_sources = list(dict.fromkeys(c["source"] for c in chunks_raw if c["source"]))
    return chunks_raw, deduped, unique_sources


async def execute_search_docs(
    tool_args: dict,
    collection: str,
) -> str:
    query = tool_args.get("query", "")
    catalog_refs = tool_args.get("catalog_refs") or []

    if catalog_refs:
        refs_str = " | ".join(catalog_refs)
        query = (
            f"Élément recherché : {query} "
            f"| Postes catalogue à vérifier : {refs_str}"
        )
        logger.info("search_docs: query enriched with %d catalog_refs", len(catalog_refs))

    logger.info("search_docs query=%r collection=%r", query, collection)
    try:
        from backend.api.dependencies import get_collection_manager
        from core.search import RAGEngine

        cm = get_collection_manager()
        rag = RAGEngine(
            nom_collection=collection,
            prompt_name="defaut",
            collection_manager=cm,
        )
        contexte, sources = await asyncio.to_thread(rag.rechercher, query)
        context_parts = [c for c in contexte.split("\n\n---\n\n") if c.strip()]

        chunks_raw, deduped, unique_sources = await process_chunks_and_extract(
            context_parts, sources, query, log_prefix="search_docs",
        )
        extraction = {"components": deduped, "sources": unique_sources}
        logger.info(
            "search_docs: %d chunk(s) → %d composant(s), sources: %s",
            len(chunks_raw), len(deduped), unique_sources,
        )
        return json.dumps(extraction, ensure_ascii=False)
    except Exception as exc:
        logger.warning("search_docs failed: %s", exc)
        return json.dumps({"error": str(exc)})


async def execute_report_findings(tool_args: dict) -> str:
    components = tool_args.get("components", [])
    context = tool_args.get("context", "")
    return json.dumps({"found": components, "context": context})


async def execute_search_collection_excel(
    tool_args: dict,
    collection: str,
    excel_adapter,
) -> str:
    query = tool_args.get("query", "")
    logger.info("search_collection_excel query=%r collection=%r", query, collection)
    if excel_adapter is None:
        return json.dumps({"error": "ExcelCollectionAdapter non disponible"})
    if not excel_adapter.has_excel_data(collection):
        return json.dumps({"error": "Aucun fichier Excel dans cette collection", "results": []})
    try:
        sql_result = await asyncio.to_thread(excel_adapter.nl2sql_query, query, collection)
        logger.info(
            "search_collection_excel → %d résultats, SQL: %s",
            len(sql_result.get("results", [])), sql_result.get("sql", "")[:200],
        )
        return json.dumps(sql_result, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.warning("search_collection_excel failed: %s", exc)
        return json.dumps({"error": str(exc)})
