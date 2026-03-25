"""
Catalog tools: search_catalog, add_to_panier, update_panier_item, plan_tasks.
Extracted from devis_service.py — same logic, standalone async functions.
"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_CATALOG_PATH = Path("/app/documents/catalogue.xlsx")
_CATALOG_CHALLENGE_COLLECTION = "_catalog_challenge"


def _s(value) -> str:
    """Cast any catalog field value to str safely (handles int/float/None)."""
    return str(value).strip() if value is not None else ""


def _detect_element_conflict(query: str, raw_results: list[dict]) -> dict | None:
    """
    From raw BM25 results, find rows where elements (col H) match the query.
    Returns a choices event dict for element-level results, or None.
    """
    q_words = re.findall(r"\w{3,}", query.lower())
    if not q_words:
        return None

    element_groups: dict[str, list[dict]] = defaultdict(list)
    for row in raw_results:
        elements_text = _s(row.get("elements"))
        if not elements_text:
            continue
        el_words = re.findall(r"\w+", elements_text.lower())
        for qw in q_words:
            if any(ew.startswith(qw) or qw.startswith(ew) for ew in el_words):
                element_groups[elements_text].append(row)
                break

    if not element_groups:
        return None

    def _occ(row: dict, el_text: str) -> dict:
        price = row.get("fourniture")
        return {
            "action":      "add_element",
            "elements":    el_text,
            "nom_poste":   _s(row.get("nom_poste")),
            "num_poste":   _s(row.get("num_poste")),
            "nom_affaire": _s(row.get("nom_affaire")),
            "num_affaire": _s(row.get("num_affaire")),
            "fournisseur": _s(row.get("fournisseur")),
            "fourniture":  str(price) if price is not None else "",
            "ensemble":    _s(row.get("ensemble")),
        }

    el_data: list[dict] = []
    for el_text, rows in element_groups.items():
        seen: set[tuple] = set()
        occs: list[dict] = []
        for row in rows:
            key = (_s(row.get("num_affaire")), _s(row.get("num_poste")))
            if key not in seen:
                seen.add(key)
                occs.append(_occ(row, el_text))
        el_data.append({"text": el_text, "keys": seen, "occs": occs})

    def _poste_option(occ: dict) -> dict:
        detail = f"Affaire : {occ['nom_affaire']}"
        if occ["num_poste"]:
            detail += f" · Poste n°{occ['num_poste']}"
        return {
            "id": json.dumps({
                "action":      "add_poste",
                "nom_poste":   occ["nom_poste"],
                "nom_affaire": occ["nom_affaire"],
                "num_poste":   occ["num_poste"],
            }, ensure_ascii=False),
            "label":  occ["nom_poste"],
            "detail": detail,
        }

    def _alone_option(el: dict) -> dict:
        aff_labels = [o["nom_affaire"] for o in el["occs"] if o["nom_affaire"]]
        detail = ("Affaires : " if len(aff_labels) > 1 else "Affaire : ") + ", ".join(aff_labels) if aff_labels else ""
        if len(el["occs"]) == 1:
            id_data: dict = {**el["occs"][0]}
        else:
            id_data = {"action": "show_element_affaires", "element_text": el["text"], "occurrences": el["occs"]}
        return {"id": json.dumps(id_data, ensure_ascii=False), "label": "Ajouter l'élément seul", "detail": detail}

    if len(el_data) == 1:
        el = el_data[0]
        seen_pk: set[tuple] = set()
        options: list[dict] = []
        for occ in el["occs"]:
            pk = (occ["nom_poste"], occ["nom_affaire"])
            if occ["nom_poste"] and pk not in seen_pk:
                seen_pk.add(pk)
                options.append(_poste_option(occ))
        options.append(_alone_option(el))
        return {
            "type": "element",
            "question": f"L'élément « {el['text']} » appartient à ce(s) poste(s). Que souhaitez-vous ajouter ?",
            "options": options,
        }

    common_keys: set[tuple] = el_data[0]["keys"].copy()
    for el in el_data[1:]:
        common_keys &= el["keys"]

    elements_label = ", ".join(f"« {el['text']} »" for el in el_data)

    if common_keys:
        options = []
        seen_pk = set()
        for num_aff, num_pos in common_keys:
            for el in el_data:
                for occ in el["occs"]:
                    if occ["num_affaire"] == num_aff and occ["num_poste"] == num_pos:
                        pk = (occ["nom_poste"], occ["nom_affaire"])
                        if pk not in seen_pk:
                            seen_pk.add(pk)
                            options.append(_poste_option(occ))
                        break
                else:
                    continue
                break
        sep = [{"text": el["text"], "occurrences": el["occs"]} for el in el_data]
        options.append({
            "id":    json.dumps({"action": "show_elements", "elements": sep}, ensure_ascii=False),
            "label": "Ajouter les éléments séparément",
        })
        return {
            "type": "element",
            "question": f"Les éléments {elements_label} appartiennent au(x) même(s) poste(s). Que souhaitez-vous faire ?",
            "options": options,
        }

    options = []
    for el in el_data:
        aff_labels = [o["nom_affaire"] for o in el["occs"] if o["nom_affaire"]]
        detail = ("Affaires : " if len(aff_labels) > 1 else "Affaire : ") + ", ".join(aff_labels) if aff_labels else ""
        if len(el["occs"]) == 1:
            id_data = {**el["occs"][0]}
        else:
            id_data = {"action": "show_element_affaires", "element_text": el["text"], "occurrences": el["occs"]}
        options.append({"id": json.dumps(id_data, ensure_ascii=False), "label": el["text"], "detail": detail})
    return {
        "type": "element",
        "question": f"Aucun poste commun pour {elements_label}. Lequel des éléments souhaitez-vous ajouter ?",
        "options": options,
    }


def _build_column_choice(query: str) -> dict:
    return {
        "type": "search_column",
        "question": (
            f"Aucun poste « {query} » trouvé dans le catalogue. "
            "Chercher dans une autre colonne ?"
        ),
        "options": [
            {
                "id": json.dumps({"action": "search_column", "column": "elements", "query": query}, ensure_ascii=False),
                "label": "Sous-composant",
                "detail": "Chercher parmi les éléments internes (col H)",
            },
            {
                "id": json.dumps({"action": "search_column", "column": "ensemble", "query": query}, ensure_ascii=False),
                "label": "Type d'ensemble",
                "detail": "Ex : Mécanique, Électrique, Pneumatique",
            },
            {
                "id": json.dumps({"action": "search_column", "column": "nom_affaire", "query": query}, ensure_ascii=False),
                "label": "Affaire (projet)",
                "detail": "Chercher par nom de projet",
            },
            {
                "id": json.dumps({"action": "search_column", "column": "fournisseur", "query": query}, ensure_ascii=False),
                "label": "Fournisseur",
                "detail": "Chercher par nom de fournisseur",
            },
            {
                "id": json.dumps({"action": "refine"}, ensure_ascii=False),
                "label": "Affiner ma recherche",
                "detail": "Reformuler votre demande",
            },
        ],
    }


def _build_relevance_choice(query: str, results: list[dict]) -> dict:
    poste_rows: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        nom_poste = _s(row.get("nom_poste"))
        if nom_poste:
            poste_rows[nom_poste].append(row)

    options: list[dict] = []
    for nom_poste, rows in list(poste_rows.items())[:3]:
        seen_occs: set[tuple] = set()
        occurrences: list[dict] = []
        for row in rows:
            occ_key = (_s(row.get("nom_affaire")), _s(row.get("num_poste")))
            if occ_key not in seen_occs:
                seen_occs.add(occ_key)
                occurrences.append({
                    "nom_affaire": _s(row.get("nom_affaire")),
                    "num_poste":   _s(row.get("num_poste")),
                    "fournisseur": _s(row.get("fournisseur")),
                    "ensemble":    _s(row.get("ensemble")),
                })
        row = rows[0]
        detail_parts = []
        if row.get("ensemble"):
            detail_parts.append(f"Ensemble : {row['ensemble']}")
        if row.get("fournisseur"):
            detail_parts.append(f"Fournisseur : {row['fournisseur']}")
        options.append({
            "id": json.dumps({"nom_poste": nom_poste, "occurrences": occurrences}, ensure_ascii=False),
            "label": f"Utiliser : {nom_poste}",
            "detail": " · ".join(detail_parts) or None,
        })

    options.append({
        "id": json.dumps({"action": "search_docs", "query": query}, ensure_ascii=False),
        "label": "Chercher dans la documentation PDF",
        "detail": "Rechercher les composants dans les fiches techniques",
    })
    options.append({
        "id": json.dumps({"action": "refine"}, ensure_ascii=False),
        "label": "Affiner la recherche",
        "detail": "Reformuler votre demande",
    })

    return {
        "type": "relevance",
        "question": (
            f"Aucun résultat ne correspond directement à « {query} ». "
            "Que souhaitez-vous faire ?"
        ),
        "options": options,
    }


def _detect_conflict(results: list[dict], current_affaire: str | None) -> dict | None:
    if not results:
        return None

    poste_rows: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        nom_poste = _s(row.get("nom_poste"))
        if nom_poste:
            poste_rows[nom_poste].append(row)

    distinct_postes = list(poste_rows.keys())

    if current_affaire and len(distinct_postes) <= 1:
        return None

    if len(distinct_postes) > 1:
        options = []
        for nom_poste, rows in poste_rows.items():
            seen_occs: set[tuple] = set()
            occurrences: list[dict] = []
            for row in rows:
                occ_key = (_s(row.get("nom_affaire")), _s(row.get("num_poste")))
                if occ_key not in seen_occs:
                    seen_occs.add(occ_key)
                    occurrences.append({
                        "nom_affaire": _s(row.get("nom_affaire")),
                        "num_poste":   _s(row.get("num_poste")),
                        "fournisseur": _s(row.get("fournisseur")),
                        "ensemble":    _s(row.get("ensemble")),
                    })
            row = rows[0]
            detail_parts = []
            if row.get("ensemble"):
                detail_parts.append(f"Ensemble : {row['ensemble']}")
            options.append({
                "id": json.dumps({"nom_poste": nom_poste, "occurrences": occurrences}, ensure_ascii=False),
                "label": nom_poste,
                "detail": " · ".join(detail_parts),
            })
        return {
            "type": "poste",
            "question": "Plusieurs postes correspondent à votre recherche. Lequel souhaitez-vous ajouter ?",
            "options": options,
        }

    if len(distinct_postes) == 1:
        nom_poste = distinct_postes[0]
        rows = poste_rows[nom_poste]
        distinct_affaires = list({
            _s(r.get("nom_affaire"))
            for r in rows
            if r.get("nom_affaire") is not None
        })
        if len(distinct_affaires) > 1:
            seen: set[str] = set()
            options = []
            for row in rows:
                aff = _s(row.get("nom_affaire"))
                if aff and aff not in seen:
                    seen.add(aff)
                    prix_total = row.get("prix_total")
                    detail = f"Poste : {nom_poste}"
                    if prix_total is not None:
                        detail += f" · Prix total : {round(float(prix_total), 2)}€"
                    options.append({"id": aff, "label": aff, "detail": detail})
            if len(options) > 1:
                return {
                    "type": "affaire",
                    "question": f"Le poste « {nom_poste} » existe dans plusieurs affaires. Quelle affaire utiliser pour ce devis ?",
                    "nom_poste": nom_poste,
                    "options": options,
                }

    return None


async def execute_search_catalog(
    tool_args: dict,
    conversation_id: str,
    catalog,
    excel_adapter,
    catalog_method: str = "bm25",
    _ctx: dict | None = None,
) -> str:
    query = tool_args.get("query", "")
    column = tool_args.get("column", "nom_poste")
    logger.info("search_catalog query=%r catalog_method=%r", query, catalog_method)

    _use_sql = catalog_method == "sql" and excel_adapter is not None
    if _use_sql:
        if not excel_adapter.has_excel_data(_CATALOG_CHALLENGE_COLLECTION):
            if _CATALOG_PATH.exists():
                logger.info("search_catalog (SQL): indexation du catalogue…")
                await asyncio.to_thread(excel_adapter.ingest, _CATALOG_PATH, _CATALOG_CHALLENGE_COLLECTION)
            else:
                logger.warning("search_catalog (SQL): catalogue introuvable, fallback BM25")
                _use_sql = False

    if _use_sql and excel_adapter.has_excel_data(_CATALOG_CHALLENGE_COLLECTION):
        sql_result = await asyncio.to_thread(excel_adapter.nl2sql_query, query, _CATALOG_CHALLENGE_COLLECTION)
        results = sql_result.get("results", [])
        logger.info("search_catalog (SQL) → %d résultats bruts, SQL: %s", len(results), sql_result.get("sql", "")[:200])
        if results:
            filtered = await asyncio.to_thread(catalog.filter_to_column_matches, query, column, results)
            if filtered:
                results = filtered
                logger.info("search_catalog (SQL) → %d résultats après filtre colonne=%r", len(results), column)
    else:
        results = await asyncio.to_thread(catalog.search, query, 40, column)
        logger.info("search_catalog (BM25) → %d results (raw)", len(results))

    current_affaire = await asyncio.to_thread(catalog.get_current_affaire, conversation_id)
    search_all = await asyncio.to_thread(catalog.get_search_scope, conversation_id)
    if current_affaire and not search_all:
        filtered = [r for r in results if _s(r.get("nom_affaire")).lower() == current_affaire.lower().strip()]
        if filtered:
            results = filtered
            logger.info("search_catalog filtered to affaire=%r → %d results", current_affaire, len(filtered))

    if _ctx is not None:
        _ctx["last_raw_results"] = results

    logger.info("search_catalog: column=%r BM25 → %d résultats", column, len(results))

    seen_pairs: set[tuple] = set()
    pairs: list[tuple[str, str]] = []
    for row in results:
        nom_p = (row.get("nom_poste") or "").strip()
        if not nom_p:
            continue
        na = (row.get("nom_affaire") or "").strip().lower()
        key = (nom_p, na)
        if key not in seen_pairs:
            seen_pairs.add(key)
            pairs.append((nom_p, na))

    deduped = await asyncio.to_thread(catalog.get_postes_aggregated_by_name, pairs)
    pair_order = {p: i for i, p in enumerate(pairs)}
    deduped.sort(key=lambda r: pair_order.get(
        ((r.get("nom_poste") or "").strip(), (r.get("nom_affaire") or "").strip().lower()),
        999,
    ))
    logger.info("search_catalog aggregated → %d postes distincts", len(deduped))

    if len(deduped) == 1:
        deduped[0]["_hint"] = "1 résultat → add_to_panier maintenant."
    elif len(deduped) > 1:
        deduped[0]["_next_action"] = "N postes → appelle ask_user_choice maintenant."

    return json.dumps(deduped, ensure_ascii=False, default=str)


async def execute_plan_tasks(tool_args: dict, conversation_id: str, catalog) -> str:
    items = tool_args.get("items", [])
    if isinstance(items, list):
        items = [str(i).strip() for i in items if str(i).strip()]
    if items:
        await asyncio.to_thread(catalog.set_task_list, conversation_id, items)
        logger.info("plan_tasks: liste enregistrée → %r", items)
        return json.dumps({"planned": items})
    return json.dumps({"error": "liste vide"})


async def execute_add_to_panier(tool_args: dict, conversation_id: str, catalog) -> str:
    postes = tool_args.get("postes", [])
    if isinstance(postes, str):
        try:
            postes = json.loads(postes)
        except json.JSONDecodeError:
            postes = []
    if isinstance(postes, dict):
        postes = [postes]

    search_all = await asyncio.to_thread(catalog.get_search_scope, conversation_id)
    if search_all and postes:
        first = postes[0] if isinstance(postes[0], dict) else {}
        new_affaire = (first.get("nom_affaire") or "").strip()
        if new_affaire:
            await asyncio.to_thread(catalog.set_affaire_lock, conversation_id, new_affaire)

    result = await asyncio.to_thread(catalog.add_to_panier, conversation_id, postes)
    added  = result.get("added", [])
    errors = result.get("errors", [])
    current_affaire = result.get("current_affaire")

    for added_poste in added:
        await asyncio.to_thread(catalog.complete_task_item, conversation_id, added_poste["nom_poste"])

    response: dict = {
        "added": len(added),
        "postes": [p["nom_poste"] for p in added],
        "current_affaire": current_affaire,
    }
    if errors:
        response["errors"] = errors
        affaire_mismatches = [e for e in errors if e.get("error") == "affaire_mismatch"]
        if affaire_mismatches:
            response["warning"] = (
                f"Certains postes ont été rejetés car ils appartiennent à une affaire différente "
                f"({affaire_mismatches[0]['requested_affaire']}) alors que le devis utilise "
                f"l'affaire '{current_affaire}'. Informe l'utilisateur."
            )
    if added:
        await asyncio.to_thread(catalog.set_search_scope, conversation_id, False)

    return json.dumps(response, ensure_ascii=False, default=str)


async def execute_update_panier_item(tool_args: dict, conversation_id: str, catalog) -> str:
    nom_poste = tool_args.get("nom_poste", "").strip()
    panier = await asyncio.to_thread(catalog.get_panier, conversation_id)
    matched = next((p for p in panier if p["nom_poste"] == nom_poste), None)
    if not matched:
        matched = next((p for p in panier if p["nom_poste"].lower() == nom_poste.lower()), None)
    if not matched:
        matched = next((p for p in panier if nom_poste.lower() in p["nom_poste"].lower() or p["nom_poste"].lower() in nom_poste.lower()), None)
    if not matched:
        return json.dumps({"error": f"Poste '{nom_poste}' non trouvé dans le panier"})

    fields: dict = {}
    for f in ("nbre_jours_etude", "nbre_jours_atelier", "nbre_jours_client"):
        if f in tool_args:
            fields[f] = int(tool_args[f])
    if "is_option" in tool_args:
        fields["is_option"] = bool(tool_args["is_option"])
    if not fields:
        return json.dumps({"error": "Aucun champ à modifier"})

    await asyncio.to_thread(catalog.update_panier_item, conversation_id, matched["id"], fields)
    logger.info("update_panier_item: %s → %r", matched["nom_poste"], fields)
    await asyncio.to_thread(catalog.complete_task_item, conversation_id, matched["nom_poste"])
    return json.dumps({"updated": matched["nom_poste"], "item_id": matched["id"], "fields": fields})
