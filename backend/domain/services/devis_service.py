"""
Devis service: orchestrates LLM tool-calling loop for quote generation.

Flow per chat turn
──────────────────
1. Non-streaming Ollama call with tool definitions attached.
2. If the response contains tool_calls → execute them, emit SSE status events,
   append results to messages, loop (max MAX_TOOL_ITERATIONS).
3. When no more tool_calls → streaming Ollama call with full context
   (no tools passed, so the LLM generates plain text).
4. Emit {"done": true, "panier": [...]} at the end.

SSE event shapes
────────────────
{"token": "..."}                         — streaming text token
{"tool_call": {"name": "...", "status": "running|done"}}
{"panier": [...]}                        — panier state snapshot
{"choices": {"question": "...", "options": [...]}}  — user choice cards
{"done": true, "panier": [...]}          — end of stream
{"error": "..."}                         — error
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MAX_TOOL_ITERATIONS = 12  # safety cap on the tool-calling loop (RFQ complexe = plan_tasks + N searches)

# ── Structured output : JSON schemas pour Ollama format-constrained generation ──
# Grammar-based constrainte → le LLM retourne TOUJOURS du JSON valide.
# Élimine les re.search / json.loads fragiles et les "pas de JSON dans la réponse".
_SCHEMA_DIMENSIONS = {
    "type": "object",
    "properties": {
        "dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string"},
                    "query":     {"type": "string"},
                },
                "required": ["dimension", "query"],
            },
        }
    },
    "required": ["dimensions"],
}
_SCHEMA_COMPONENTS = {
    "type": "object",
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nom":   {"type": "string"},
                    "specs": {"type": "string"},
                },
                "required": ["nom", "specs"],
            },
        }
    },
    "required": ["components"],
}

# ── Context compression (tool-calling loop) ─────────────────────────────────────
# Au-delà de ce seuil (chars excl. system prompt), les messages tool accumulés
# sont résumés en un bloc compact avant le prochain appel LLM.
# Objectif : rester sous ~2500 tokens pour éviter la troncature Ollama 4096→8192.
_COMPRESS_CHARS_THRESHOLD = 7000

_CATALOG_PATH = Path("/app/documents/catalogue.xlsx")
_CATALOG_CHALLENGE_COLLECTION = "_catalog_challenge"

def _build_system_prompt(
    collection: str,
    current_coefficient: float = 0.0,
    current_coef_final: float = 0.0,
    panier: list[dict] | None = None,
    task_list: list[dict] | None = None,
    rfq_context: str | None = None,
) -> str:
    if panier:
        lines = [
            f"  • {p['nom_poste']} | affaire : {p.get('nom_affaire') or '?'} | "
            f"Étude:{p.get('nbre_jours_etude', 0)}j Atelier:{p.get('nbre_jours_atelier', 0)}j Client:{p.get('nbre_jours_client', 0)}j"
            f"{' | EN OPTION' if p.get('is_option') else ''}"
            for p in panier
        ]
        panier_section = "Postes déjà dans le devis :\n" + "\n".join(lines)
    else:
        panier_section = "Le panier est vide."

    if task_list:
        import re as _re
        # Cross-reference with panier: a task is done if its query matches any panier item
        panier_noms_lower = {(p.get("nom_poste") or "").lower() for p in (panier or [])}
        def _task_done(task: dict) -> bool:
            if task.get("done"):
                return True
            q_words = set(_re.findall(r"\w{3,}", task["query"].lower()))
            return any(
                bool(q_words & set(_re.findall(r"\w{3,}", nom)))
                for nom in panier_noms_lower
            )
        task_lines = [
            f"  {'[✓]' if _task_done(t) else '[ ]'} {t['query']}"
            for t in task_list
        ]
        task_section = "LISTE DE TÂCHES :\n" + "\n".join(task_lines)
        pending = [t for t in task_list if not _task_done(t)]
        next_task = f"\n  → Prochain item à chercher : \"{pending[0]['query']}\"" if pending else "\n  → Tous les items sont ajoutés."
        task_section += next_task
    else:
        task_section = "LISTE DE TÂCHES : (vide — utilise plan_tasks si plusieurs postes demandés)"

    rfq_section = (
        f"\nCONTEXTE RFQ (analyse documentaire préalable — LECTURE SEULE) :\n{rfq_context}\n"
        "↳ Ces informations viennent des documents PDF/Excel. Elles NE contiennent PAS de num_poste catalogue.\n"
        "↳ Pour ajouter un composant identifié ici : search_catalog obligatoire d'abord.\n"
        if rfq_context
        else ""
    )

    return f"""Tu es un assistant de devis pour VLM Robotics.

CATALOGUE : historique de projets réels. Chaque poste (nom_poste) a un num_poste et appartient à une nom_affaire.

PANIER ACTUEL :
{panier_section}

{task_section}
{rfq_section}
COEFFICIENTS : fournitures={current_coefficient}% · final={current_coef_final}%

━━━ QUE FAIRE ━━━

ÉTAPE 1 — Toujours commencer par un tool call, jamais du texte.
• Demande contient "Sélectionné :" → add_to_panier directement.
• Tâche [ ] en attente → search_catalog ou update_panier_item selon la tâche.
• Message "[SYSTÈME]" ou "Cherche X" → search_catalog(query=X).
• Sinon → search_catalog pour ce que l'utilisateur demande.

ÉTAPE 2 — Après search_catalog :
• Champ "_hint" présent → add_to_panier immédiatement.
• Champ "_next_action" présent → suis l'instruction du champ (tool call obligatoire, pas de texte).
• Plusieurs postes retournés, pas de _hint → ask_user_choice avec les postes trouvés.

ÉTAPE 3 — Après search_docs :
• Appelle report_findings avec les noms de composants du résultat. Pas de texte entre les deux.

RÉPONSE TEXTE : 1 phrase max. Jamais de code, jamais de liste inventée.

━━━ RÈGLES ━━━
• nom_poste, nom_affaire, num_poste : copier les valeurs EXACTES des résultats search_catalog.
• "en option" → is_option=true. Jours mentionnés → nbre_jours_etude/atelier/client.
• Données tabulaires (tarifs, tableaux Excel) → search_collection_excel.
• set_devis_settings uniquement si l'utilisateur demande à changer les coefficients.
• UNE tâche à la fois."""

# ── Tool definitions ──────────────────────────────────────────────────────────
# Descriptions volontairement concises : le LLM n'a pas besoin de savoir QUAND
# utiliser chaque tool (c'est dans le system prompt + les hints _next_action).
# Il a besoin de savoir CE QUE fait le tool et quels paramètres passer.
# Économie estimée : ~400 tokens vs l'ancienne version.
#
# NOTE migration 5090 : avec Qwen3.5-35B-A3B (contexte 32K), l'économie de tokens
# est moins critique. Mais des descriptions courtes = moins de confusion pour le LLM.

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan_tasks",
            "description": "Enregistre la liste des actions à effectuer. Chaque item = label court.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ex: [\"search vireur\", \"update orbiteur atelier=5\"]",
                    }
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_panier_item",
            "description": "Modifie un poste existant dans le panier (jours ou option).",
            "parameters": {
                "type": "object",
                "properties": {
                    "nom_poste": {
                        "type": "string",
                        "description": "Nom EXACT du poste (copier depuis PANIER ACTUEL)",
                    },
                    "nbre_jours_etude":   {"type": "integer", "description": "Jours d'étude"},
                    "nbre_jours_atelier": {"type": "integer", "description": "Jours d'atelier"},
                    "nbre_jours_client":  {"type": "integer", "description": "Jours client"},
                    "is_option": {"type": "boolean", "description": "true = option"},
                },
                "required": ["nom_poste"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Recherche dans le catalogue VLM. Retourne des postes avec prix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Terme de recherche",
                    },
                    "column": {
                        "type": "string",
                        "enum": ["nom_poste", "elements", "ensemble", "nom_affaire", "fournisseur"],
                        "description": "Colonne cible (défaut: nom_poste)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_collection_excel",
            "description": "Recherche dans les Excel de la collection (tarifs, tableaux). Pas le catalogue VLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question ou terme de recherche",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_findings",
            "description": "Liste les composants trouvés dans les docs. Appeler après search_docs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Noms spécifiques de modèles/références trouvés (ex: 'KR360', 'Fanuc R-2000iC')",
                    },
                    "context": {
                        "type": "string",
                        "description": "Résumé court de ce qui a été trouvé",
                    },
                    "catalog_matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "nom_poste": {"type": "string"},
                                "spec_status": {
                                    "type": "string",
                                    "enum": ["match", "partial", "no_match", "unknown"],
                                },
                                "note": {"type": "string"},
                            },
                            "required": ["nom_poste", "spec_status"],
                        },
                        "description": "Statut specs par poste catalogue (si catalog_refs utilisé)",
                    },
                    "doc_only_models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Modèles dans les docs mais absents du catalogue",
                    },
                    "suggestion": {
                        "type": "string",
                        "description": "Recommandation courte",
                    },
                },
                "required": ["components"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Recherche dans les PDFs techniques. Retourne composants + specs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Type de composant + contraintes (ex: 'vireur 8T basculeur')",
                    },
                    "catalog_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Noms exacts des postes catalogue à vérifier dans les docs",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_choice",
            "description": "Présente des choix cliquables à l'utilisateur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":     {"type": "string"},
                                "label":  {"type": "string"},
                                "detail": {"type": "string"},
                            },
                            "required": ["id", "label"],
                        },
                    },
                },
                "required": ["question", "options"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_devis_settings",
            "description": "Met à jour les coefficients du devis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coefficient": {
                        "type": "number",
                        "description": "Coefficient fournitures en %",
                    },
                    "coef_final": {
                        "type": "number",
                        "description": "Coefficient final en %",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_panier",
            "description": "Ajoute des postes au panier. Utiliser les valeurs EXACTES du catalogue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "postes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "nom_poste":   {"type": "string", "description": "Valeur EXACTE de nom_poste"},
                                "nom_affaire": {"type": "string", "description": "Valeur EXACTE de nom_affaire"},
                                "num_poste":   {"type": "string"},
                                "ensemble":    {"type": "string"},
                                "quantite":    {"type": "integer"},
                                "nbre_jours_etude":   {"type": "integer"},
                                "nbre_jours_atelier": {"type": "integer"},
                                "nbre_jours_client":  {"type": "integer"},
                                "is_option":          {"type": "boolean"},
                            },
                            "required": ["nom_poste", "nom_affaire"],
                        },
                    }
                },
                "required": ["postes"],
            },
        },
    },
]


def _s(value) -> str:
    """Cast any catalog field value to str safely (handles int/float/None)."""
    return str(value).strip() if value is not None else ""


def _query_matches_postes(query: str, results: list[dict]) -> bool:
    """
    True if the query semantically targets a POSTE name (col G) rather than an element.
    Delegates to CatalogAdapter.query_matches_postes which uses the BM25 Snowball tokenizer,
    so plurals like 'orbiteurs' correctly match 'Orbiteur VLM V500'.
    """
    # Fast path: delegate to catalog adapter at call site (needs catalog instance).
    # This standalone version uses a simple regex word intersection as fallback.
    import re as _re
    q_words = _re.findall(r"\w{3,}", query.lower())
    if not q_words:
        return bool(results)
    for row in results:
        nom_poste = _s(row.get("nom_poste")).lower()
        np_words = _re.findall(r"\w+", nom_poste)
        for qw in q_words:
            for npw in np_words:
                if npw.startswith(qw) or qw.startswith(npw):
                    return True
    return False


def _detect_element_conflict(query: str, raw_results: list[dict]) -> dict | None:
    """
    From raw BM25 results, find rows where elements (col H) match the query.

    For each distinct element text, collects all (num_affaire, num_poste) occurrences,
    then computes the intersection to find postes containing ALL requested elements.

    Cases:
    • 1 element  → parent postes (add_poste) + element alone option
    • N elements, common poste(s) → common postes (add_poste) + add separately option
    • N elements, no common poste → one card per element (add_element / show_element_affaires)

    Every option id is JSON with an 'action' field; the frontend dispatches accordingly.
    """
    import re as _re
    from collections import defaultdict

    q_words = _re.findall(r"\w{3,}", query.lower())
    if not q_words:
        return None

    # ── Step 1: group rows by element text ──────────────────────────────────
    element_groups: dict[str, list[dict]] = defaultdict(list)
    for row in raw_results:
        elements_text = _s(row.get("elements"))
        if not elements_text:
            continue
        el_words = _re.findall(r"\w+", elements_text.lower())
        for qw in q_words:
            if any(ew.startswith(qw) or qw.startswith(ew) for ew in el_words):
                element_groups[elements_text].append(row)
                break

    if not element_groups:
        return None

    # ── Step 2: per element, build unique occurrences by (num_affaire, num_poste) ─
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

    # ── Helpers ───────────────────────────────────────────────────────────────
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

    # ── Case A: single element ────────────────────────────────────────────────
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

    # ── Cases B & C: multiple elements — compute intersection ─────────────────
    common_keys: set[tuple] = el_data[0]["keys"].copy()
    for el in el_data[1:]:
        common_keys &= el["keys"]

    elements_label = ", ".join(f"« {el['text']} »" for el in el_data)

    # ── Case B: common poste(s) found ─────────────────────────────────────────
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

    # ── Case C: no common poste — one card per element ────────────────────────
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
    """
    Emitted when no nom_poste matches the query.
    Lets the user pick which catalog column to search next.
    """
    return {
        "type": "search_column",
        "question": (
            f"Aucun poste « {query} » trouvé dans le catalogue. "
            "Chercher dans une autre colonne ?"
        ),
        "options": [
            {
                "id": json.dumps(
                    {"action": "search_column", "column": "elements", "query": query},
                    ensure_ascii=False,
                ),
                "label": "Sous-composant",
                "detail": "Chercher parmi les éléments internes (col H)",
            },
            {
                "id": json.dumps(
                    {"action": "search_column", "column": "ensemble", "query": query},
                    ensure_ascii=False,
                ),
                "label": "Type d'ensemble",
                "detail": "Ex : Mécanique, Électrique, Pneumatique",
            },
            {
                "id": json.dumps(
                    {"action": "search_column", "column": "nom_affaire", "query": query},
                    ensure_ascii=False,
                ),
                "label": "Affaire (projet)",
                "detail": "Chercher par nom de projet",
            },
            {
                "id": json.dumps(
                    {"action": "search_column", "column": "fournisseur", "query": query},
                    ensure_ascii=False,
                ),
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
    """
    Called when BM25 returned results but none semantically match the query
    (neither postes nor elements). Shows the user three types of actions:
    - One button per found nom_poste (same JSON structure as PRIORITY 1 so the
      frontend can reuse the 'poste' handler).
    - "Search in docs" — tells the LLM to call search_docs.
    - "Refine search" — lets the user rephrase.
    """
    from collections import defaultdict

    # Group by nom_poste and collect occurrences (same logic as PRIORITY 1)
    poste_rows: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        nom_poste = _s(row.get("nom_poste"))
        if nom_poste:
            poste_rows[nom_poste].append(row)

    options: list[dict] = []

    for nom_poste, rows in list(poste_rows.items())[:3]:  # cap at 3 poste options
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
            "id": json.dumps(
                {"nom_poste": nom_poste, "occurrences": occurrences},
                ensure_ascii=False,
            ),
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
    """
    Detect ambiguity in search_catalog results and return a choices event dict, or None.

    Results are already deduplicated by (nom_poste, nom_affaire) before this call,
    so each row represents one distinct poste+affaire combination.

    Priority 1 — Multiple distinct postes: show poste selection cards first.
    Priority 2 — Single poste, multiple affaires: show affaire conflict cards.
    """
    from collections import defaultdict

    if not results:
        return None

    # Group rows by nom_poste first — need distinct_postes before deciding to skip.
    poste_rows: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        nom_poste = _s(row.get("nom_poste"))
        if nom_poste:
            poste_rows[nom_poste].append(row)

    distinct_postes = list(poste_rows.keys())

    # Affaire locked + single unambiguous poste → no conflict to show.
    # Multiple postes always trigger cards (even with a locked affaire) so the
    # user never gets free-form LLM text with potentially wrong names/prices.
    if current_affaire and len(distinct_postes) <= 1:
        return None


    # ── PRIORITY 1: Multiple distinct postes ──────────────────────────────────
    # Each card's `id` embeds ALL (nom_affaire, num_poste) occurrences as JSON so
    # the frontend can show affaire sub-cards directly without a new LLM search.
    if len(distinct_postes) > 1:
        options = []
        for nom_poste, rows in poste_rows.items():
            # Collect all unique (nom_affaire, num_poste) occurrences for this poste
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

            row = rows[0]  # representative row for display
            detail_parts = []
            if row.get("ensemble"):
                detail_parts.append(f"Ensemble : {row['ensemble']}")
            options.append({
                "id": json.dumps(
                    {"nom_poste": nom_poste, "occurrences": occurrences},
                    ensure_ascii=False,
                ),
                "label": nom_poste,
                "detail": " · ".join(detail_parts),
            })
        return {
            "type": "poste",
            "question": "Plusieurs postes correspondent à votre recherche. Lequel souhaitez-vous ajouter ?",
            "options": options,
        }

    # ── PRIORITY 2: Single poste, multiple affaires ───────────────────────────
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
                    options.append({
                        "id": aff,
                        "label": aff,
                        "detail": detail,
                    })
            if len(options) > 1:
                return {
                    "type": "affaire",
                    "question": f"Le poste « {nom_poste} » existe dans plusieurs affaires. Quelle affaire utiliser pour ce devis ?",
                    "nom_poste": nom_poste,
                    "options": options,
                }

    return None


class DevisService:
    def __init__(self, catalog_adapter, excel_adapter=None) -> None:
        self.catalog = catalog_adapter
        self.excel_adapter = excel_adapter

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _process_chunks_and_extract(
        self,
        context_parts: list[str],
        sources: list[dict],
        query: str,
        log_prefix: str = "chunks",
    ) -> tuple[list[dict], list[dict], list[str]]:
        """
        Shared chunk processing: split context into chunks, extract components
        via LLM, deduplicate by name.

        Returns (chunks_raw, deduped_components, unique_sources).
        """
        MAX_CHUNKS = 3
        MAX_CHARS_PER_CHUNK = 6000
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
            extraction = await self._extract_components_from_chunks([chunk], query)
            all_components.extend(extraction.get("components", []))

        # Dedup by name (first occurrence kept)
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

    async def _ollama_chat(
        self, messages: list[dict], tools: list | None = None
    ) -> dict:
        """Non-streaming Ollama /api/chat call. Used for the tool-calling loop."""
        payload: dict = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            # think=True : le LLM raisonne avant chaque décision d'outil.
            # Le bloc <think> va dans response["message"]["thinking"] (champ séparé d'Ollama),
            # PAS dans "content" — donc il n'est PAS ajouté au message history (ligne 2151-2157).
            # Zéro pollution du contexte, meilleure qualité de décision.
            "think": True,
            "options": {"num_ctx": 8192},  # évite la troncature (default=4096 pour qwen3.5:4b)
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Log du thinking pour debug (non injecté dans le contexte)
        thinking = data.get("message", {}).get("thinking", "")
        if thinking:
            logger.info(
                "[tool_loop] 🧠 thinking (%d chars):\n%s",
                len(thinking), thinking[:1000],
            )
        return data

    async def _extract_components_from_chunks(
        self, chunks: list[dict], original_query: str
    ) -> dict:
        """
        Appel LLM léger (num_ctx=2048, temp=0) pour extraire les noms de composants
        et leurs specs depuis les chunks de documentation RAG.
        Retourne {"components": [{"nom": "...", "specs": "..."}]}.
        Fallback sur liste vide si extraction échoue.
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
        logger.info(
            "[extract_components] appel LLM — %d chunk(s), ~%d chars input",
            len(chunks), total_chars,
        )

        # format=_SCHEMA_COMPONENTS : grammar-constrained → JSON valide garanti
        # Élimine les re.search fragiles et les parse failures.
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "think": False,
            "format": _SCHEMA_COMPONENTS,
            "options": {"temperature": 0, "num_ctx": 8192},  # aligné sur tool_loop — évite reload KV cache
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

            content = data.get("message", {}).get("content", "").strip()
            logger.info("[extract_components] réponse brute (%d chars): %r", len(content), content[:500])

            parsed = json.loads(content)  # garanti valide par format schema
            # Le LLM peut retourner {"components": [...]} ou directement [...]
            if isinstance(parsed, list):
                components = parsed
            else:
                components = parsed.get("components", [])
            noms = [c.get("nom", "?") for c in components if isinstance(c, dict)]
            logger.info(
                "[extract_components] ✓ %d composant(s): %s",
                len(components), noms,
            )
            return {"components": components}

        except json.JSONDecodeError as exc:
            # Ne devrait pas arriver avec format schema — log pour debug
            logger.error(
                "[extract_components] JSONDecodeError inattendu (format schema actif): %s | raw=%r",
                exc, content[:300] if "content" in dir() else "N/A",
            )
            return {"components": [], "error": f"json_decode: {exc}"}
        except Exception as exc:
            logger.warning(
                "[extract_components] extraction échouée: %s", exc, exc_info=True
            )
            return {"components": [], "error": str(exc)}

    # ── RFQ Planner : Iterative Gap-Detection Loop ─────────────────────────────

    async def _decompose_rfq(self, message: str) -> list[dict]:
        """
        Phase 1 : décompose le message en dimensions de recherche (think=True).
        Retourne [{"dimension": "...", "query": "..."}, ...] — min 3, max 6.
        """
        system = (
            "Tu es un analyseur de RFQ industriel. "
            "Décompose le message en dimensions de recherche technique indépendantes.\n"
            "Chaque dimension = un aspect distinct à chercher dans la documentation technique.\n"
            "• Minimum 3, maximum 6 dimensions.\n"
            "• 'dimension' : nom court (ex: 'Effecteur DED Laser poudre').\n"
            "• 'query' : requête documentaire IMPÉRATIVEMENT avec les termes EXACTS du RFQ "
            "(noms de modèles, références produits, acronymes, marques) — NE PAS les remplacer "
            "par des synonymes génériques. Ex: si le RFQ dit 'SOLO', la query doit contenir 'SOLO'.\n\n"
            "Réponds UNIQUEMENT en JSON, sans texte avant ou après :\n"
            '[{"dimension": "...", "query": "..."}, ...]'
        )
        user_msg = f"RFQ à analyser :\n\n{message}"

        logger.info("[rfq_planner] Phase 1 — décomposition (%d chars)", len(message))

        # format=_SCHEMA_DIMENSIONS : grammar-constrained → {"dimensions": [...]} garanti
        # think=False : le grammar-constrained JSON + think=True causait des timeouts (bloc
        # <think> trop long avant de produire le JSON contraint → httpx.ReadTimeout à 180s).
        # La contrainte de schéma suffit pour garantir la structure — pas besoin de thinking.
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "think": False,
            "format": _SCHEMA_DIMENSIONS,
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

            content = data.get("message", {}).get("content", "").strip()
            logger.info(
                "[rfq_planner] _decompose_rfq réponse brute (%d chars): %r",
                len(content), content[:600],
            )

            parsed = json.loads(content)  # garanti valide par format schema
            # Le modèle peut retourner {"dimensions": [...]} ou directement [...]
            if isinstance(parsed, list):
                dimensions = parsed
            else:
                dimensions = parsed.get("dimensions", [])
            valid = [
                d for d in dimensions
                if isinstance(d, dict) and d.get("dimension") and d.get("query")
            ][:6]
            logger.info(
                "[rfq_planner] ✓ %d dimension(s) extraite(s): %s",
                len(valid), [d["dimension"] for d in valid],
            )
            for d in valid:
                logger.info(
                    "[rfq_planner]   • %r → query=%r",
                    d["dimension"], d["query"],
                )
            return valid

        except json.JSONDecodeError as exc:
            logger.error(
                "[rfq_planner] _decompose_rfq JSONDecodeError (format schema actif!): %s | raw=%r",
                exc, content[:300] if "content" in dir() else "N/A",
            )
            return []
        except Exception as exc:
            logger.warning("[rfq_planner] _decompose_rfq failed: %s", exc, exc_info=True)
            return []

    async def _search_and_extract_dimension(
        self, dimension: dict, collection: str
    ) -> dict:
        """
        Phase 2 : recherche RAG pour une dimension + extraction LLM.
        Réutilise _extract_components_from_chunks existant.
        """
        query = dimension.get("query", "")
        dim_name = dimension.get("dimension", "?")

        logger.info(
            "[rfq_planner] Phase 2 — dim %r | query=%r | collection=%r",
            dim_name, query, collection,
        )
        try:
            from backend.api.dependencies import get_collection_manager
            from core.search import RAGEngine

            cm = get_collection_manager()
            rag = RAGEngine(
                nom_collection=collection,
                prompt_name="defaut",
                collection_manager=cm,
                hyde_mode="composition",  # BOM-style HyDE → cible les pages de composition/fournitures
            )
            contexte, sources = await asyncio.to_thread(rag.rechercher, query)
            context_parts = [c for c in contexte.split("\n\n---\n\n") if c.strip()]

            chunks_raw, deduped, unique_sources = await self._process_chunks_and_extract(
                context_parts, sources, query, log_prefix=f"rfq_planner:{dim_name}",
            )
            nb = len(deduped)
            logger.info(
                "[rfq_planner] dim %r → %d composant(s) (%d chunks traités), sources: %s",
                dim_name, nb, len(chunks_raw), unique_sources,
            )
            return {
                "dimension": dim_name,
                "query": query,
                "components": deduped,
                "sources": unique_sources,
                # Chunks bruts conservés pour le debug endpoint (/devis/rfq-debug)
                "chunks": chunks_raw,
            }
        except Exception as exc:
            logger.warning(
                "[rfq_planner] _search_and_extract_dimension %r failed: %s",
                dim_name, exc, exc_info=True,
            )
            return {
                "dimension": dim_name,
                "query": query,
                "components": [],
                "sources": [],
                "error": str(exc),
            }

    async def _detect_rfq_gaps(
        self, original_message: str, all_findings: list[dict]
    ) -> list[dict]:
        """
        Phase 3 : identifie les aspects du RFQ non couverts par les recherches.
        Retourne jusqu'à 3 nouvelles dimensions à chercher (peut retourner []).
        """
        findings_summary = ""
        for f in all_findings:
            comp_names = [c.get("nom", "?") for c in f.get("components", [])]
            summary = ", ".join(comp_names) if comp_names else "rien trouvé"
            findings_summary += f"• {f['dimension']} : {summary}\n"

        logger.info(
            "[rfq_planner] Phase 3 — gap detection | %d dimensions cherchées:\n%s",
            len(all_findings), findings_summary,
        )

        system = (
            "Tu es un analyseur de RFQ. Identifie les aspects importants du RFQ "
            "qui ne sont PAS encore couverts par les résultats de recherche.\n"
            "Réponds avec une liste JSON (max 3 éléments, peut être vide []) :\n"
            '[{"dimension": "...", "query": "..."}]\n'
            "Retourne [] si tout est couvert ou si les gaps ne sont pas critiques."
        )
        user_msg = (
            f"RFQ original :\n{original_message}\n\n"
            f"Ce qui a été trouvé dans la documentation :\n{findings_summary}\n\n"
            "Quels aspects critiques du RFQ manquent dans les résultats ?"
        )
        # Réutilise _SCHEMA_DIMENSIONS (même structure dimension+query)
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "think": False,
            "format": _SCHEMA_DIMENSIONS,
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

            content = data.get("message", {}).get("content", "").strip()
            logger.info(
                "[rfq_planner] _detect_rfq_gaps réponse brute (%d chars): %r",
                len(content), content[:400],
            )

            parsed = json.loads(content)  # garanti valide
            # Le LLM peut retourner {"dimensions": [...]} ou directement [...]
            if isinstance(parsed, list):
                gaps = parsed
            else:
                gaps = parsed.get("dimensions", [])
            valid_gaps = [
                g for g in gaps
                if isinstance(g, dict) and g.get("dimension") and g.get("query")
            ][:3]

            if valid_gaps:
                logger.info(
                    "[rfq_planner] ✓ %d gap(s) identifié(s):",
                    len(valid_gaps),
                )
                for g in valid_gaps:
                    logger.info(
                        "[rfq_planner]   gap %r → query=%r",
                        g["dimension"], g["query"],
                    )
            else:
                logger.info("[rfq_planner] Phase 3 ✓ — aucun gap critique détecté")
            return valid_gaps

        except json.JSONDecodeError as exc:
            logger.error(
                "[rfq_planner] _detect_rfq_gaps JSONDecodeError (format schema actif!): %s",
                exc,
            )
            return []
        except Exception as exc:
            logger.warning("[rfq_planner] _detect_rfq_gaps failed: %s", exc, exc_info=True)
            return []

    # ── Auto-plan : server-side multi-action detection ──────────────────────────

    @staticmethod
    def _detect_multi_action(message: str) -> list[str] | None:
        """
        Détecte les demandes multi-actions dans le message utilisateur.
        Retourne une liste d'items pour plan_tasks, ou None si mono-action.

        Patterns détectés :
        - "ajoute X et Y" / "cherche X et Y"
        - "ajoute X, Y et Z"
        - "X + Y + Z"
        - "modifie X et Y"
        - Listes avec ":" suivi de tirets ou numéros

        Ne se déclenche PAS pour :
        - Messages "[SYSTÈME]" ou "Sélectionné :"
        - Messages courts (< 10 chars)
        - Messages qui ne contiennent pas de conjonction/séparateur
        """
        msg = message.strip()

        # Skip messages système et sélections utilisateur
        if msg.startswith("[SYSTÈME]") or msg.startswith("Sélectionné") or msg.startswith("Cherche"):
            return None
        if len(msg) < 10:
            return None

        # ── Pattern 1 : verbe d'action + liste avec "et" / virgules ──────
        # Exemples :
        #   "ajoute un vireur et un orbiteur"
        #   "cherche vireur, orbiteur et armoire"
        #   "ajoute vireur, orbiteur, armoire électrique"
        #   "mets 5 jours étude sur le vireur et 3 sur l'orbiteur"

        # Identifier le verbe d'action (optionnel — certains messages n'en ont pas)
        action_verbs = re.findall(
            r"^(ajoute|cherche|recherche|trouve|mets|modifie|update|supprime|passe)\b",
            msg.lower(),
        )
        action_prefix = action_verbs[0] if action_verbs else ""

        # Chercher une structure de liste :
        # "X, Y et Z" ou "X et Y" ou "X, Y, Z"
        # On split sur " et " et "," en preservant les segments
        msg_lower = msg.lower()

        # Retirer le verbe d'action du début pour isoler les items
        work_text = msg_lower
        if action_prefix:
            work_text = re.sub(r"^" + re.escape(action_prefix) + r"\s+", "", work_text)

        # Split sur " et " et ","
        # "un vireur, un orbiteur et une armoire" → ["un vireur", "un orbiteur", "une armoire"]
        parts = re.split(r"\s+et\s+|,\s*", work_text)
        parts = [p.strip() for p in parts if p.strip()]

        # Nettoyer les articles et déterminants communs
        cleaned = []
        for p in parts:
            # Retirer articles, pronoms et déterminants de début
            p = re.sub(r"^(moi|toi|lui|nous|vous|eux|une|un|les|le|la|des|d'|l[''])\s*", "", p).strip()
            # Deuxième passe : "un vireur" restant après "moi un" → retirer encore
            p = re.sub(r"^(une|un|les|le|la|des)\s+", "", p).strip()
            if p and len(p) >= 2:
                cleaned.append(p)

        if len(cleaned) < 2:
            return None

        # Construire les items de task list
        items = []
        for item_text in cleaned:
            # Détecter si c'est un update ("5 jours étude sur le vireur")
            update_match = re.match(
                r"(\d+)\s*(?:jours?\s+)?(étude|atelier|client)\s+(?:sur\s+(?:le|la|l[''])\s*)?(.+)",
                item_text,
            )
            if update_match:
                jours, type_j, poste = update_match.groups()
                field_map = {"étude": "etude", "atelier": "atelier", "client": "client"}
                field = field_map.get(type_j, type_j)
                items.append(f"update {poste.strip()} {field}={jours}")
            elif action_prefix in ("modifie", "update", "mets", "passe"):
                items.append(f"update {item_text}")
            else:
                items.append(f"search {item_text}")

        logger.info(
            "[auto_plan] message=%r → %d items détectés: %r",
            msg[:80], len(items), items,
        )
        return items if len(items) >= 2 else None

    # ── Server-side guardrails : detect when LLM should have called a tool ─────

    # Sentinel retourné par _detect_expected_tool quand un nudge est nécessaire
    # mais qu'aucun tool synthétique ne peut être construit.
    _NEEDS_NUDGE = {"_needs_nudge": True}

    def _detect_expected_tool(
        self,
        messages: list[dict],
        task_list: list[dict] | None,
        conversation_id: str,
    ) -> dict | None:
        """
        Analyse l'état courant pour déterminer si le LLM aurait dû appeler un tool
        au lieu de générer du texte.

        Retourne :
        - Un dict {"name": "...", "arguments": {...}} → tool call synthétique à exécuter
        - self._NEEDS_NUDGE → le LLM doit réessayer avec un rappel
        - None → fin normale (pas de tool attendu)

        Cas détectés :
        1. Tâches [ ] en attente dans la task list → search_catalog pour la prochaine tâche
        2. Dernier tool result contenait _next_action → le LLM devait appeler un tool
        3. Dernier tool était search_docs → report_findings attendu
        """
        # ── Cas 3 : search_docs sans report_findings ──────────────────────────
        # Parcourir les messages en ordre inverse pour trouver le dernier couple
        # (assistant tool_call → tool result). On cherche d'abord le tool result,
        # puis on remonte pour trouver quel tool l'a produit.
        last_tool_name = None
        last_tool_content = None
        _found_tool_result = False
        for m in reversed(messages):
            if not _found_tool_result and m.get("role") == "tool":
                last_tool_content = m.get("content", "")
                _found_tool_result = True
                continue
            if _found_tool_result and m.get("role") == "assistant" and m.get("tool_calls"):
                tcs = m["tool_calls"]
                if tcs:
                    last_tool_name = tcs[-1].get("function", {}).get("name")
                break

        # Si le dernier tool call était search_docs et qu'on a un result,
        # le LLM aurait dû enchaîner avec report_findings
        if last_tool_name == "search_docs" and last_tool_content:
            try:
                parsed = json.loads(last_tool_content)
                components = parsed.get("components", [])
                comp_names = [c.get("nom", "?") for c in components if isinstance(c, dict)]
                if comp_names:
                    logger.info(
                        "[guardrail] search_docs sans report_findings → injection synthétique (%d composants)",
                        len(comp_names),
                    )
                    return {
                        "name": "report_findings",
                        "arguments": {"components": comp_names},
                    }
            except (json.JSONDecodeError, AttributeError):
                pass

        # ── Cas 2 : _next_action ignoré ───────────────────────────────────────
        if last_tool_content:
            try:
                parsed = json.loads(last_tool_content)
                if isinstance(parsed, list) and parsed:
                    first = parsed[0] if isinstance(parsed[0], dict) else {}
                    if first.get("_next_action"):
                        # Le LLM aurait dû appeler ask_user_choice ou search_docs
                        # On ne peut pas deviner lequel → nudge pour réessayer
                        logger.info(
                            "[guardrail] _next_action présent mais LLM a généré du texte → nudge"
                        )
                        return self._NEEDS_NUDGE
            except (json.JSONDecodeError, AttributeError):
                pass

        # ── Cas 1 : tâches en attente ─────────────────────────────────────────
        if task_list:
            pending = [t for t in task_list if not t.get("done")]
            if pending:
                next_query = pending[0].get("query", "")
                # Distinguer search vs update
                q_lower = next_query.lower().strip()
                if q_lower.startswith("update "):
                    # update task → on ne peut pas construire un update_panier_item
                    # sans connaitre les champs exacts → nudge seulement.
                    # NOTE migration 5090 : Qwen3.5-35B-A3B (IFBench 91.5) devrait
                    # réussir le nudge systématiquement. Si ce n'est pas le cas,
                    # parser "update X champ=val" ici pour injection.
                    logger.info(
                        "[guardrail] tâche update en attente %r → nudge (pas d'injection)",
                        next_query,
                    )
                    return self._NEEDS_NUDGE
                else:
                    # search task → on peut construire un search_catalog synthétique
                    # Nettoyer le préfixe "search " s'il existe
                    search_query = q_lower.removeprefix("search ").strip()
                    if search_query:
                        logger.info(
                            "[guardrail] tâche en attente → search_catalog synthétique query=%r",
                            search_query,
                        )
                        return {
                            "name": "search_catalog",
                            "arguments": {"query": search_query, "column": "nom_poste"},
                        }

        return None  # Rien d'anormal — fin de boucle légitime

    # ── Context compression helpers ────────────────────────────────────────────

    @staticmethod
    def _estimate_chars(messages: list[dict]) -> int:
        """
        Estime la taille totale des messages hors system prompt (messages[0]).
        On exclut le system prompt car il est rebuild à chaque itération (taille fixe).
        """
        return sum(
            len(str(m.get("content", ""))) + len(str(m.get("tool_calls", "") or ""))
            for m in messages[1:]
        )

    async def _compress_tool_results(
        self, messages: list[dict], original_user_msg: str
    ) -> list[dict]:
        """
        Compresse l'historique des tool results accumulés en un résumé compact (~150 mots).
        Retourne une liste réduite : [system_prompt, compressed_summary].
        Le system prompt sera rebuild au prochain tour — il n'est pas inclus ici.

        Format de sortie :
          [Recherches précédentes résumées]
          • plan_tasks → ['search vireur', 'search orbiteur']
          • search_catalog "vireur" → Vireur VLMV3T, VLMV500, ORB100, ORB1000
          • search_docs "SOLO config" → DossierTechnique_SOLO_FR_ind3.pdf (5 chunks)
        """
        before_chars = self._estimate_chars(messages)
        nb_messages = len(messages)

        # Extraire l'historique des outils (ignorer system + premier user)
        tool_history_parts = []
        for m in messages[2:]:
            role = m.get("role", "")
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", {})
                    if isinstance(args, dict):
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in list(args.items())[:2])
                    else:
                        args_str = str(args)[:80]
                    tool_history_parts.append(f"→ appel {name}({args_str})")
            elif role == "tool":
                content_preview = str(m.get("content", ""))[:600]
                tool_history_parts.append(f"  résultat: {content_preview}")

        if not tool_history_parts:
            logger.info("[compress] Rien à compresser (pas de tool history)")
            return messages

        tool_history_text = "\n".join(tool_history_parts)
        logger.info(
            "[compress] ⚡ Compression déclenchée | %d messages | %d chars → seuil=%d",
            nb_messages, before_chars, _COMPRESS_CHARS_THRESHOLD,
        )
        logger.debug("[compress] Historique brut à compresser:\n%s", tool_history_text[:1500])

        compression_prompt = (
            "Résume ces appels d'outils et leurs résultats en une liste compacte (max 300 mots).\n"
            "Format : '• [outil] [paramètre] → [résultat bref avec noms exacts]'\n"
            "Garde les noms de postes, modèles et composants EXACTS. Supprime le reste.\n\n"
            f"Historique à résumer :\n{tool_history_text[:3000]}"
        )
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": compression_prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": 8192},  # aligné sur tool_loop — évite reload KV cache
        }
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
            summary = resp.json().get("message", {}).get("content", "").strip()

            after_chars = len(summary) + len(original_user_msg)
            ratio = round(before_chars / max(after_chars, 1), 1)
            logger.info(
                "[compress] ✓ %d chars → %d chars (ratio x%s) | résumé:\n%s",
                before_chars, after_chars, ratio, summary,
            )

            compressed_messages = [
                messages[0],  # system prompt (sera rebuild de toute façon)
                {"role": "user", "content": original_user_msg},
                {
                    "role": "assistant",
                    "content": f"[Recherches précédentes résumées]\n{summary}",
                },
            ]
            logger.info(
                "[compress] messages: %d → %d (économie: %d messages)",
                nb_messages, len(compressed_messages), nb_messages - len(compressed_messages),
            )
            return compressed_messages

        except Exception as exc:
            logger.warning(
                "[compress] ⚠ Compression échouée (%s) — messages originaux conservés", exc
            )
            return messages  # fallback : garder les messages originaux

    async def _synthesize_rfq_context(
        self, original_message: str, all_findings: list[dict]
    ) -> str:
        """
        Phase 4 : synthèse compacte (~200-400 tokens) pour enrichir le system prompt.
        Le résultat est injecté comme CONTEXTE RFQ dans _build_system_prompt.
        """
        findings_text = ""
        for f in all_findings:
            comp_names = [c.get("nom", "?") for c in f.get("components", [])]
            specs = [c.get("specs", "") for c in f.get("components", []) if c.get("specs")]
            sources = f.get("sources", [])
            findings_text += f"\n**{f['dimension']}** :\n"
            if comp_names:
                findings_text += f"  Composants : {', '.join(comp_names)}\n"
                if specs:
                    findings_text += f"  Specs : {' | '.join(specs[:2])}\n"
            else:
                findings_text += "  Aucun composant identifié dans la documentation.\n"
            if sources:
                src_names = [s.split("/")[-1] for s in sources]
                findings_text += f"  Sources : {', '.join(src_names)}\n"

        logger.info(
            "[rfq_planner] Phase 4 — synthèse de %d dimension(s):\n%s",
            len(all_findings), findings_text[:800],
        )

        system = (
            "Tu es un synthétiseur de résultats de recherche technique. "
            "Crée un contexte compact (max 250 mots) pour aider un assistant devis "
            "à traiter une RFQ industrielle.\n"
            "Format : bullet points par dimension, composants trouvés avec specs clés.\n"
            "Sois concis et factuel. Utilise uniquement les informations fournies."
        )
        user_msg = (
            f"RFQ :\n{original_message[:500]}\n\n"
            f"Résultats de recherche documentaire :\n{findings_text}\n\n"
            "Synthétise en contexte de devis."
        )
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

            synthesis = data.get("message", {}).get("content", "").strip()
            logger.info(
                "[rfq_planner] synthèse produite (%d chars):\n%s",
                len(synthesis), synthesis[:600],
            )
            return synthesis
        except Exception as exc:
            logger.warning("[rfq_planner] _synthesize_rfq_context failed: %s", exc)
            # Fallback : synthèse directe depuis les findings sans LLM
            lines = []
            for f in all_findings:
                comp_names = [c.get("nom", "?") for c in f.get("components", [])]
                lines.append(
                    f"• {f['dimension']} : "
                    + (", ".join(comp_names) if comp_names else "non trouvé")
                )
            return "\n".join(lines)

    async def _run_rfq_planner(
        self, message: str, collection: str
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator : Iterative Gap-Detection Loop pour analyser un RFQ complexe.

        4 phases :
        1. ANALYZING   : décomposition LLM générique (think=True, min 3 max 6 dims)
        2. SEARCHING   : RAG search + extraction LLM par dimension
        3. GAP_CHECK   : détection des lacunes (1 cycle, max 3 gaps)
        4. SYNTHESIZING: synthèse compacte pour enrichir le system prompt devis

        Yields {"rfq_planning": {"status": "...", "step": "...", ...}} events.
        Dernier event : {"rfq_planning": {"status": "done", "context": "...", "dimensions_found": N}}
        """
        logger.info("=" * 60)
        logger.info("[rfq_planner] ══ DÉMARRAGE RFQPlanner ══")
        logger.info("[rfq_planner] message: %d chars | collection: %r", len(message), collection)
        logger.info("=" * 60)

        # Vérifier que la collection existe avant de lancer les appels LLM
        try:
            from backend.api.dependencies import get_collection_manager
            _cm = get_collection_manager()
            if not _cm.collection_existe(collection):
                logger.warning(
                    "[rfq_planner] collection %r inexistante — planner ignoré", collection
                )
                yield {"rfq_planning": {"status": "done", "context": "", "dimensions_found": 0}}
                return
        except Exception as exc:
            logger.warning("[rfq_planner] vérif. collection échouée: %s — poursuite", exc)

        all_findings: list[dict] = []

        # ── Phase 1 : Décomposition ──────────────────────────────────────────
        yield {"rfq_planning": {"status": "analyzing", "step": "Analyse du RFQ en cours…"}}
        logger.info("[rfq_planner] ── Phase 1 : décomposition")

        dimensions = await self._decompose_rfq(message)

        if not dimensions:
            logger.warning("[rfq_planner] 0 dimensions extraites — abandon du planner")
            yield {"rfq_planning": {"status": "done", "context": "", "dimensions_found": 0}}
            return

        logger.info("[rfq_planner] Phase 1 ✓ — %d dimensions", len(dimensions))
        yield {
            "rfq_planning": {
                "status": "analyzing",
                "step": f"{len(dimensions)} dimensions identifiées",
                "dimensions_found": len(dimensions),
            }
        }

        # ── Phase 2 : Recherche RAG — séquentielle (Ollama single-GPU) ──────
        # asyncio.gather parallèle → queue Ollama saturée → timeouts en cascade
        # Sur RTX 5090 / vLLM multi-instance → repasser en gather
        logger.info(
            "[rfq_planner] ── Phase 2 : %d dimensions séquentielles (Ollama single-GPU)",
            len(dimensions),
        )
        for d in dimensions:
            logger.info(
                "[rfq_planner]   • %r → query=%r",
                d.get("dimension", "?"), d.get("query", ""),
            )
        yield {
            "rfq_planning": {
                "status": "searching",
                "step": f"Recherche {len(dimensions)} dimensions…",
                "dimensions_found": len(dimensions),
            }
        }

        import time as _time
        _phase2_start = _time.monotonic()
        results = []
        for dim in dimensions:
            try:
                result = await self._search_and_extract_dimension(dim, collection)
            except Exception as exc:
                result = exc
            results.append(result)
        _phase2_elapsed = _time.monotonic() - _phase2_start

        all_findings = []
        total_components = 0
        for i, result in enumerate(results):
            dim_name = dimensions[i].get("dimension", f"Dim {i + 1}")
            if isinstance(result, Exception):
                logger.warning(
                    "[rfq_planner] [%d/%d] %r — exception: %s",
                    i + 1, len(dimensions), dim_name, result,
                )
                all_findings.append({
                    "dimension": dim_name,
                    "query": dimensions[i].get("query", ""),
                    "components": [],
                    "sources": [],
                    "error": str(result),
                })
            else:
                nb = len(result.get("components", []))
                total_components += nb
                logger.info(
                    "[rfq_planner] [%d/%d] %r → %d composant(s) | sources: %s",
                    i + 1, len(dimensions), dim_name, nb, result.get("sources", []),
                )
                all_findings.append(result)

        logger.info(
            "[rfq_planner] Phase 2 ✓ — %d dimensions | %d composants totaux | %.1fs",
            len(all_findings), total_components, _phase2_elapsed,
        )

        # ── Phase 3 : Gap Detection (1 cycle max) ────────────────────────────
        logger.info("[rfq_planner] ── Phase 3 : détection des gaps")
        yield {
            "rfq_planning": {
                "status": "gap_check",
                "step": "Vérification des lacunes…",
                "dimensions_found": len(all_findings),
            }
        }

        gaps = await self._detect_rfq_gaps(message, all_findings)

        if gaps:
            logger.info("[rfq_planner] Phase 3 : %d gap(s) → recherche complémentaire", len(gaps))
            for j, gap in enumerate(gaps):
                gap_name = gap.get("dimension", f"Gap {j + 1}")
                logger.info(
                    "[rfq_planner] Gap [%d/%d]: %r | query: %r",
                    j + 1, len(gaps), gap_name, gap.get("query", ""),
                )
                yield {
                    "rfq_planning": {
                        "status": "searching",
                        "step": f"Complément [{j + 1}/{len(gaps)}] : {gap_name}",
                        "dimensions_found": len(dimensions) + len(gaps),
                    }
                }
                gap_finding = await self._search_and_extract_dimension(gap, collection)
                all_findings.append(gap_finding)
                nb = len(gap_finding.get("components", []))
                logger.info(
                    "[rfq_planner] Gap [%d/%d] → %d composant(s)", j + 1, len(gaps), nb
                )
        else:
            logger.info("[rfq_planner] Phase 3 ✓ — aucun gap critique détecté")

        # ── Phase 4 : Synthèse ───────────────────────────────────────────────
        logger.info("[rfq_planner] ── Phase 4 : synthèse du contexte")
        yield {
            "rfq_planning": {
                "status": "synthesizing",
                "step": "Synthèse du contexte RFQ…",
                "dimensions_found": len(all_findings),
            }
        }

        rfq_context = await self._synthesize_rfq_context(message, all_findings)

        # ── Phase SQL : catalogue lookup pour construire rfq_candidates ───────
        logger.info("[rfq_planner] ── Phase SQL : lookup catalogue par composant")
        candidates_by_key: dict[str, dict] = {}  # keyed by nom_poste.lower()
        for finding in all_findings:
            dim_name = finding.get("dimension", "?")
            for comp in finding.get("components", []):
                comp_nom = comp.get("nom", "").strip()
                if not comp_nom:
                    continue
                try:
                    rows = await asyncio.to_thread(
                        self.catalog.search_hybrid, comp_nom, 8, "nom_poste"
                    )
                    for row in rows:
                        nom_poste = (row.get("nom_poste") or "").strip()
                        if not nom_poste:
                            continue
                        key = nom_poste.lower()
                        score = float(row.get("_score", 0.0))
                        # Keep the occurrence with the best score
                        if key not in candidates_by_key or score > candidates_by_key[key].get("_score", 0.0):
                            candidates_by_key[key] = {
                                "nom_poste":         nom_poste,
                                "nom_affaire":       row.get("nom_affaire", ""),
                                "num_poste":         row.get("num_poste", ""),
                                "ensemble":          row.get("ensemble", ""),
                                "fournisseur":       row.get("fournisseur", ""),
                                "prix_unitaire":     row.get("prix_unitaire"),
                                "_score":            score,
                                "_confidence":       row.get("_confidence", "fts_only"),
                                "dimension":         dim_name,
                                "composant_source":  comp_nom,
                            }
                except Exception as exc:
                    logger.warning(
                        "[rfq_planner] SQL lookup failed for %r: %s", comp_nom, exc
                    )

        rfq_candidates = sorted(
            candidates_by_key.values(), key=lambda r: r.get("_score", 0.0), reverse=True
        )
        logger.info("[rfq_planner] rfq_candidates: %d postes uniques", len(rfq_candidates))

        if rfq_candidates:
            yield {"rfq_candidates": rfq_candidates}

        total_comp = sum(len(f.get("components", [])) for f in all_findings)
        logger.info("=" * 60)
        logger.info("[rfq_planner] ══ TERMINÉ ══")
        logger.info(
            "[rfq_planner] %d dimensions | %d composants | contexte %d chars",
            len(all_findings), total_comp, len(rfq_context),
        )
        logger.info("[rfq_planner] CONTEXTE FINAL :\n%s", rfq_context)
        logger.info("=" * 60)

        yield {
            "rfq_planning": {
                "status": "done",
                "step": f"Contexte établi — {len(all_findings)} dimensions",
                "dimensions_found": len(all_findings),
                "context": rfq_context,
            }
        }

    async def run_rfq_planner_debug(self, message: str, collection: str) -> dict:
        """
        Version debug du RFQPlanner : exécute les 4 phases et retourne une structure
        JSON complète avec les chunks bruts, composants extraits et contexte final.
        Utilisé par le endpoint GET /api/v1/devis/rfq-debug.
        """
        import time as _time

        result: dict = {
            "phases": {
                "decomposition": {"dimensions": [], "error": None},
                "search": {"findings": []},
                "gaps": {"detected": [], "findings": []},
                "synthesis": {"rfq_context": ""},
            },
            "timing_s": {},
        }

        # ── Phase 1 ───────────────────────────────────────────────────────────
        t0 = _time.monotonic()
        try:
            dimensions = await self._decompose_rfq(message)
        except Exception as exc:
            result["phases"]["decomposition"]["error"] = str(exc)
            dimensions = []
        result["phases"]["decomposition"]["dimensions"] = dimensions
        result["timing_s"]["decompose"] = round(_time.monotonic() - t0, 2)

        if not dimensions:
            return result

        # ── Phase 2 ───────────────────────────────────────────────────────────
        t0 = _time.monotonic()
        search_findings = []
        for dim in dimensions:
            try:
                finding = await self._search_and_extract_dimension(dim, collection)
            except Exception as exc:
                finding = {
                    "dimension": dim.get("dimension", "?"),
                    "query": dim.get("query", ""),
                    "components": [],
                    "sources": [],
                    "chunks": [],
                    "error": str(exc),
                }
            search_findings.append(finding)
        result["phases"]["search"]["findings"] = search_findings
        result["timing_s"]["search"] = round(_time.monotonic() - t0, 2)

        all_findings = list(search_findings)

        # ── Phase 3 ───────────────────────────────────────────────────────────
        t0 = _time.monotonic()
        try:
            gaps = await self._detect_rfq_gaps(message, all_findings)
        except Exception as exc:
            gaps = []
            result["phases"]["gaps"]["error"] = str(exc)
        result["phases"]["gaps"]["detected"] = gaps

        gap_findings = []
        for gap in gaps:
            try:
                gf = await self._search_and_extract_dimension(gap, collection)
            except Exception as exc:
                gf = {
                    "dimension": gap.get("dimension", "?"),
                    "query": gap.get("query", ""),
                    "components": [],
                    "sources": [],
                    "chunks": [],
                    "error": str(exc),
                }
            gap_findings.append(gf)
            all_findings.append(gf)
        result["phases"]["gaps"]["findings"] = gap_findings
        result["timing_s"]["gaps"] = round(_time.monotonic() - t0, 2)

        # ── Phase SQL : lookup catalogue pour chaque composant extrait ───────────
        t0 = _time.monotonic()
        for finding in all_findings:
            sql_postes: list[dict] = []
            for comp in finding.get("components", []):
                nom = comp.get("nom", "").strip()
                if not nom:
                    continue
                try:
                    rows = await asyncio.to_thread(self.catalog.search_hybrid, nom, 8, "nom_poste")
                    for row in rows:
                        sql_postes.append({
                            "composant_nom": nom,
                            "nom_poste": row.get("nom_poste", ""),
                            "nom_affaire": row.get("nom_affaire", ""),
                            "ensemble": row.get("ensemble", ""),
                            "fournisseur": row.get("fournisseur", ""),
                            "prix_unitaire": row.get("prix_unitaire"),
                            "_score": row.get("_score", 0.0),
                            "_confidence": row.get("_confidence", "fts_only"),
                        })
                except Exception as exc:
                    logger.warning("[rfq_debug] SQL lookup failed for %r: %s", nom, exc)
            finding["sql_postes"] = sql_postes
        result["timing_s"]["sql_lookup"] = round(_time.monotonic() - t0, 2)

        # ── Contexte structuré brut (sans LLM) ────────────────────────────────
        structured_lines: list[str] = []
        for finding in all_findings:
            dim = finding.get("dimension", "?")
            comp_names = [c.get("nom", "?") for c in finding.get("components", [])]
            sql_postes = finding.get("sql_postes", [])
            structured_lines.append(f"[{dim}]")
            if comp_names:
                structured_lines.append(f"  Composants identifiés : {', '.join(comp_names)}")
            if sql_postes:
                seen = set()
                for p in sql_postes:
                    key = p["nom_poste"]
                    if key in seen:
                        continue
                    seen.add(key)
                    affaire = f" ({p['nom_affaire']})" if p.get("nom_affaire") else ""
                    prix = f" — {p['prix_unitaire']}€" if p.get("prix_unitaire") else ""
                    structured_lines.append(f"  • {p['nom_poste']}{affaire}{prix}")
            else:
                structured_lines.append("  (aucun poste catalogue trouvé)")
            structured_lines.append("")
        structured_context = "\n".join(structured_lines).strip()
        result["phases"]["synthesis"]["structured_context"] = structured_context

        # ── Phase 4 ───────────────────────────────────────────────────────────
        t0 = _time.monotonic()
        try:
            rfq_context = await self._synthesize_rfq_context(message, all_findings)
        except Exception as exc:
            rfq_context = f"(erreur synthèse: {exc})"
        result["phases"]["synthesis"]["rfq_context"] = rfq_context
        result["timing_s"]["synthesis"] = round(_time.monotonic() - t0, 2)

        return result

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
        collection: str,
        conversation_id: str,
        catalog_method: str = "bm25",
    ) -> str:
        """Execute a single tool call and return result as JSON string."""
        if tool_name == "plan_tasks":
            items = tool_args.get("items", [])
            if isinstance(items, list):
                items = [str(i).strip() for i in items if str(i).strip()]
            if items:
                await asyncio.to_thread(
                    self.catalog.set_task_list, conversation_id, items
                )
                logger.info("plan_tasks: liste enregistrée → %r", items)
                return json.dumps({"planned": items})
            return json.dumps({"error": "liste vide"})

        if tool_name == "search_catalog":
            query = tool_args.get("query", "")
            logger.info("search_catalog query=%r catalog_method=%r", query, catalog_method)
            column = tool_args.get("column", "nom_poste")

            # ── SQL (NL2SQL) path ──────────────────────────────────────────────
            _use_sql = catalog_method == "sql" and self.excel_adapter is not None
            if _use_sql:
                if not self.excel_adapter.has_excel_data(_CATALOG_CHALLENGE_COLLECTION):
                    if _CATALOG_PATH.exists():
                        logger.info("search_catalog (SQL): indexation du catalogue…")
                        await asyncio.to_thread(
                            self.excel_adapter.ingest, _CATALOG_PATH, _CATALOG_CHALLENGE_COLLECTION
                        )
                    else:
                        logger.warning("search_catalog (SQL): catalogue introuvable, fallback BM25")
                        _use_sql = False

            if _use_sql and self.excel_adapter.has_excel_data(_CATALOG_CHALLENGE_COLLECTION):
                sql_result = await asyncio.to_thread(
                    self.excel_adapter.nl2sql_query, query, _CATALOG_CHALLENGE_COLLECTION
                )
                results = sql_result.get("results", [])
                logger.info(
                    "search_catalog (SQL) → %d résultats bruts, SQL: %s",
                    len(results), sql_result.get("sql", "")[:200],
                )
                # FTS5 cherche dans toutes les colonnes → post-filtrer sur la colonne cible
                # pour ne garder que les lignes où la colonne demandée contient réellement la requête
                if results:
                    filtered = await asyncio.to_thread(
                        self.catalog.filter_to_column_matches, query, column, results
                    )
                    if filtered:
                        results = filtered
                        logger.info(
                            "search_catalog (SQL) → %d résultats après filtre colonne=%r",
                            len(results), column,
                        )
            else:
                # ── BM25 path (default) ────────────────────────────────────────
                results = await asyncio.to_thread(self.catalog.search, query, 40, column)
                logger.info("search_catalog (BM25) → %d results (raw)", len(results))

            # Fetch current affaire context and user's chosen search scope
            current_affaire = await asyncio.to_thread(
                self.catalog.get_current_affaire, conversation_id
            )
            search_all = await asyncio.to_thread(
                self.catalog.get_search_scope, conversation_id
            )
            if current_affaire and not search_all:
                # Scope restricted to locked affaire (default behaviour)
                filtered = [
                    r for r in results
                    if _s(r.get("nom_affaire")).lower() == current_affaire.lower().strip()
                ]
                if filtered:
                    results = filtered
                    logger.info(
                        "search_catalog filtered to affaire=%r → %d results",
                        current_affaire, len(filtered),
                    )
            elif current_affaire and search_all:
                logger.info(
                    "search_catalog: browse_all mode — catalogue complet retourné (%d résultats)",
                    len(results),
                )

            # Stash results for element detection in chat_stream.
            self._last_raw_results: list[dict] = results

            # Per-column BM25 already searched the right column — no post-filter needed.
            # If BM25 returned nothing, results is empty → column choice card will be emitted.
            logger.info(
                "search_catalog: column=%r BM25 → %d résultats",
                column, len(results),
            )

            # Extract unique (nom_poste, nom_affaire) pairs from BM25 results.
            # Using nom_poste (not num_poste) as key: num_poste is not unique per poste
            # within an affaire in practice, which caused unrelated postes to appear.
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

            # SQL GROUP BY + SUM(fourniture) → prix_total per poste+affaire.
            deduped = await asyncio.to_thread(
                self.catalog.get_postes_aggregated_by_name, pairs
            )

            # Re-apply BM25 ranking order (SQL GROUP BY doesn't preserve it).
            pair_order = {p: i for i, p in enumerate(pairs)}
            deduped.sort(key=lambda r: pair_order.get(
                ((r.get("nom_poste") or "").strip(),
                 (r.get("nom_affaire") or "").strip().lower()),
                999,
            ))
            logger.info("search_catalog aggregated → %d postes distincts", len(deduped))

            if len(deduped) == 1:
                # Single unambiguous result: tell LLM to add immediately.
                deduped[0]["_hint"] = "1 résultat → add_to_panier maintenant."
            elif len(deduped) > 1:
                # Multiple postes: demander à l'utilisateur lequel il veut.
                # Le LLM appelle ask_user_choice. Pas de texte, pas de search_docs.
                deduped[0]["_next_action"] = "N postes → appelle ask_user_choice maintenant."

            return json.dumps(deduped, ensure_ascii=False, default=str)

        if tool_name == "search_collection_excel":
            query = tool_args.get("query", "")
            logger.info("search_collection_excel query=%r collection=%r", query, collection)
            if self.excel_adapter is None:
                return json.dumps({"error": "ExcelCollectionAdapter non disponible"})
            if not self.excel_adapter.has_excel_data(collection):
                return json.dumps({"error": "Aucun fichier Excel dans cette collection", "results": []})
            try:
                sql_result = await asyncio.to_thread(
                    self.excel_adapter.nl2sql_query, query, collection
                )
                logger.info(
                    "search_collection_excel → %d résultats, SQL: %s",
                    len(sql_result.get("results", [])), sql_result.get("sql", "")[:200],
                )
                return json.dumps(sql_result, ensure_ascii=False, default=str)
            except Exception as exc:
                logger.warning("search_collection_excel failed: %s", exc)
                return json.dumps({"error": str(exc)})

        if tool_name == "report_findings":
            components = tool_args.get("components", [])
            context = tool_args.get("context", "")
            return json.dumps({"found": components, "context": context})

        if tool_name == "search_docs":
            query = tool_args.get("query", "")
            catalog_refs = tool_args.get("catalog_refs") or []
            # Enrich the RAG query with catalog refs using a structured format so the
            # multi-query generator understands the search intent (verify specs on specific
            # models) rather than treating the refs as part of the component description.
            if catalog_refs:
                refs_str = " | ".join(catalog_refs)
                query = (
                    f"Élément recherché : {query} "
                    f"| Postes catalogue à vérifier : {refs_str}"
                )
                logger.info("search_docs: query enriched with %d catalog_refs", len(catalog_refs))
            # Always use the collection from the request — never trust the LLM's choice
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
                # Pipeline complet : HyDE + multi-query + BM25 + reranker.
                # USE_PARENT_CHILD=true → retourne parent text (~800-1200 tokens)
                # pour avoir le contexte complet des tableaux de specs.
                contexte, sources = await asyncio.to_thread(rag.rechercher, query)
                context_parts = [c for c in contexte.split("\n\n---\n\n") if c.strip()]

                chunks_raw, deduped, unique_sources = await self._process_chunks_and_extract(
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

        if tool_name == "add_to_panier":
            postes = tool_args.get("postes", [])
            # LLM sometimes sends postes as a JSON string instead of a list
            if isinstance(postes, str):
                try:
                    postes = json.loads(postes)
                except json.JSONDecodeError:
                    postes = []
            # LLM sometimes sends a single dict instead of a list
            if isinstance(postes, dict):
                postes = [postes]
            # If search_all is active (user chose "Tout le catalogue"), update the
            # affaire lock to the new affaire before inserting so the consistency
            # check in add_to_panier passes.
            search_all = await asyncio.to_thread(
                self.catalog.get_search_scope, conversation_id
            )
            if search_all and postes:
                first = postes[0] if isinstance(postes[0], dict) else {}
                new_affaire = (first.get("nom_affaire") or "").strip()
                if new_affaire:
                    await asyncio.to_thread(
                        self.catalog.set_affaire_lock, conversation_id, new_affaire
                    )
            result = await asyncio.to_thread(
                self.catalog.add_to_panier, conversation_id, postes
            )
            added  = result.get("added", [])
            errors = result.get("errors", [])
            current_affaire = result.get("current_affaire")
            # Auto-complete matching tasks in the task list
            for added_poste in added:
                await asyncio.to_thread(
                    self.catalog.complete_task_item,
                    conversation_id,
                    added_poste["nom_poste"],
                )
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
            # Reset search scope after a successful add so the next search defaults
            # to the locked affaire again (choice cards will re-ask the user).
            if added:
                await asyncio.to_thread(
                    self.catalog.set_search_scope, conversation_id, False
                )
            return json.dumps(response, ensure_ascii=False, default=str)

        if tool_name == "update_panier_item":
            nom_poste = tool_args.get("nom_poste", "").strip()
            panier = await asyncio.to_thread(self.catalog.get_panier, conversation_id)
            matched = next((p for p in panier if p["nom_poste"] == nom_poste), None)
            if not matched:
                matched = next((p for p in panier if p["nom_poste"].lower() == nom_poste.lower()), None)
            if not matched:
                matched = next((p for p in panier
                                if nom_poste.lower() in p["nom_poste"].lower()
                                or p["nom_poste"].lower() in nom_poste.lower()), None)
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
            await asyncio.to_thread(
                self.catalog.update_panier_item, conversation_id, matched["id"], fields
            )
            logger.info("update_panier_item: %s → %r", matched["nom_poste"], fields)
            await asyncio.to_thread(
                self.catalog.complete_task_item, conversation_id, matched["nom_poste"]
            )
            return json.dumps({"updated": matched["nom_poste"], "item_id": matched["id"], "fields": fields})

        if tool_name == "set_devis_settings":
            coefficient = float(tool_args.get("coefficient", -1))
            coef_final  = float(tool_args.get("coef_final", -1))
            # Read current values to fill in the one not supplied
            current = await asyncio.to_thread(self.catalog.get_devis_settings, conversation_id)
            if coefficient < 0:
                coefficient = current.get("coefficient", 0.0)
            if coef_final < 0:
                coef_final = current.get("coef_final", 0.0)
            await asyncio.to_thread(
                self.catalog.set_devis_settings, conversation_id, coefficient, coef_final
            )
            return json.dumps({"coefficient": coefficient, "coef_final": coef_final})

        return json.dumps({"error": f"Outil inconnu : {tool_name}"})

    # ── Public API ─────────────────────────────────────────────────────────────

    async def chat_stream(
        self,
        message: str,
        collection: str,
        conversation_id: str,
        history: list[dict],
        catalog_method: str = "bm25",
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator yielding SSE events for a devis chat turn.
        Handles the tool-calling loop then streams the final text response.
        """
        settings = await asyncio.to_thread(self.catalog.get_devis_settings, conversation_id)
        current_panier = await asyncio.to_thread(self.catalog.get_panier, conversation_id)
        current_tasks = await asyncio.to_thread(self.catalog.get_task_list, conversation_id)

        rfq_context: str | None = None
        _add_to_panier_succeeded = False

        # ── Auto-plan : détecter les demandes multi-actions côté serveur ────
        # Évite de dépendre du LLM pour appeler plan_tasks.
        # Le pattern matching ci-dessous détecte les conjonctions ("et", virgules)
        # dans les messages qui demandent d'ajouter/chercher/modifier plusieurs postes.
        # NOTE migration 5090 : Qwen3.5-35B-A3B (BFCL 66.1) appellera probablement
        # plan_tasks tout seul. Le auto-plan reste un filet — on pourra le désactiver
        # si le modèle est fiable sur ce point.
        if not current_tasks:
            auto_items = self._detect_multi_action(message)
            if auto_items:
                logger.info("[auto_plan] détection multi-actions → %r", auto_items)
                await asyncio.to_thread(
                    self.catalog.set_task_list, conversation_id, auto_items
                )
                current_tasks = await asyncio.to_thread(
                    self.catalog.get_task_list, conversation_id
                )

        # Auto-plan depuis findings confirmation : "[SYSTÈME] Composants validés : "X", "Y"..."
        # Le frontend envoie ce message quand l'utilisateur valide les composants trouvés dans les docs.
        # _detect_multi_action saute les messages [SYSTÈME] intentionnellement → extraction manuelle ici.
        if not current_tasks and "[SYSTÈME] Composants validés :" in message:
            _cv_comps = re.findall(r'"([^"]+)"', message)
            if len(_cv_comps) >= 2:
                logger.info("[auto_plan] composants validés → plan auto: %r", _cv_comps)
                await asyncio.to_thread(
                    self.catalog.set_task_list, conversation_id, _cv_comps
                )
                current_tasks = await asyncio.to_thread(
                    self.catalog.get_task_list, conversation_id
                )

        try:
            # ── RFQ Planner : analyse documentaire sur le premier message ──────
            # Seuil 120 chars : un RFQ est un document structuré, pas une demande courte.
            # "vireur et orbiteur" (20 chars) ne doit pas déclencher les 4 phases.
            if not history and len(message) >= 120:
                logger.info(
                    "[chat_stream] Premier message — lancement RFQPlanner collection=%r",
                    collection,
                )
                async for planner_event in self._run_rfq_planner(message, collection):
                    if (
                        "rfq_planning" in planner_event
                        and planner_event["rfq_planning"].get("status") == "done"
                    ):
                        rfq_context = planner_event["rfq_planning"].get("context") or None
                    yield planner_event
                logger.info(
                    "[chat_stream] RFQPlanner terminé — rfq_context: %d chars",
                    len(rfq_context) if rfq_context else 0,
                )

            messages: list[dict] = [{"role": "system", "content": _build_system_prompt(
                collection,
                current_coefficient=settings.get("coefficient", 0.0),
                current_coef_final=settings.get("coef_final", 0.0),
                panier=current_panier,
                task_list=current_tasks or None,
                rfq_context=rfq_context,
            )}]
            messages.extend(history)
            messages.append({"role": "user", "content": message})

            # ── Tool-calling loop (non-streaming) ─────────────────────────────
            _original_user_msg = message  # gardé pour la compression
            for _iter in range(MAX_TOOL_ITERATIONS):
                # ── Context compression : évite la troncature Ollama ──────────
                _msg_chars = self._estimate_chars(messages)
                logger.info(
                    "[tool_loop] iter=%d | messages=%d | chars(excl.sys)=%d | seuil=%d",
                    _iter, len(messages), _msg_chars, _COMPRESS_CHARS_THRESHOLD,
                )
                if _msg_chars > _COMPRESS_CHARS_THRESHOLD:
                    logger.info(
                        "[tool_loop] ⚡ Seuil dépassé (%d > %d) → compression",
                        _msg_chars, _COMPRESS_CHARS_THRESHOLD,
                    )
                    messages = await self._compress_tool_results(messages, _original_user_msg)

                # Rebuild system prompt with fresh panier + task list so the LLM
                # always sees up-to-date state (tasks checked off, new panier items).
                _panier_now = await asyncio.to_thread(self.catalog.get_panier, conversation_id)
                _tasks_now  = await asyncio.to_thread(self.catalog.get_task_list, conversation_id)
                messages[0] = {"role": "system", "content": _build_system_prompt(
                    collection,
                    current_coefficient=settings.get("coefficient", 0.0),
                    current_coef_final=settings.get("coef_final", 0.0),
                    panier=_panier_now,
                    task_list=_tasks_now or None,
                    rfq_context=rfq_context,
                )}
                _sys_chars = len(messages[0]["content"])
                logger.info(
                    "[tool_loop] system_prompt=%d chars | total_prompt~%d chars (~%d tokens)",
                    _sys_chars, _sys_chars + _msg_chars, (_sys_chars + _msg_chars) // 4,
                )
                response = await self._ollama_chat(messages, tools=_TOOLS)
                assistant_msg = response.get("message", {})
                tool_calls = assistant_msg.get("tool_calls") or []

                if not tool_calls:
                    # ── Guardrail : le LLM aurait-il dû appeler un tool ? ─────
                    # Calibré pour qwen3.5:4b (BFCL-V4 ~50, IFBench ~75).
                    # Sur RTX 5090 avec Qwen3.5-35B-A3B (BFCL 66.1, IFBench 91.5),
                    # ces guardrails se déclencheront rarement mais restent un filet
                    # de sécurité utile. Voir migration-hp-z2-rtx5090-2026-03-17.md.
                    _tasks_now_guard = await asyncio.to_thread(
                        self.catalog.get_task_list, conversation_id
                    )
                    expected = self._detect_expected_tool(
                        messages, _tasks_now_guard, conversation_id
                    )

                    if expected is not None and expected is not self._NEEDS_NUDGE:
                        # Cas avec tool call synthétique connu → exécuter directement
                        logger.info(
                            "[guardrail] injection synthétique : %s(%r)",
                            expected["name"], expected.get("arguments", {}),
                        )
                        # Ajouter le texte du LLM comme message assistant (il a quand même parlé)
                        llm_text = assistant_msg.get("content", "")
                        if llm_text:
                            messages.append({"role": "assistant", "content": llm_text})
                        # Simuler un tool call dans le flux
                        tool_calls = [{
                            "function": {
                                "name": expected["name"],
                                "arguments": expected.get("arguments", {}),
                            }
                        }]
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": tool_calls,
                        })
                        # Continuer la boucle normale ci-dessous (le for tc in tool_calls)
                    elif expected is self._NEEDS_NUDGE or (
                        _tasks_now_guard
                        and any(not t.get("done") for t in _tasks_now_guard)
                    ):
                        # Nudge : le LLM doit réessayer (soit _next_action ignoré,
                        # soit tâches en attente, soit update task non injectable)
                        if not getattr(self, "_guardrail_nudged", False):
                            self._guardrail_nudged = True
                            # Construire un nudge contextuel
                            if expected is self._NEEDS_NUDGE:
                                nudge_text = (
                                    "[SYSTÈME] Tu devais appeler un tool (ask_user_choice ou search_docs). "
                                    "Regarde le champ _next_action du dernier résultat et appelle le tool approprié."
                                )
                            else:
                                pending = [t for t in _tasks_now_guard if not t.get("done")]
                                nudge_text = (
                                    f'[SYSTÈME] Tâche en attente : "{pending[0]["query"]}". '
                                    f"Appelle le tool approprié maintenant."
                                )
                            logger.info("[guardrail] nudge → %s", nudge_text)
                            messages.append({"role": "user", "content": nudge_text})
                            continue  # re-loop pour que le LLM réessaie
                        else:
                            # Déjà nudgé une fois — si c'est _NEEDS_NUDGE (pas de tâches),
                            # on ne peut pas injecter → abandon. Si tâches, tenter injection.
                            self._guardrail_nudged = False
                            if (
                                expected is not self._NEEDS_NUDGE
                                and _tasks_now_guard
                            ):
                                # Dernière chance : injection synthétique forcée
                                pending = [t for t in _tasks_now_guard if not t.get("done")]
                                if pending:
                                    q = pending[0].get("query", "").lower().strip()
                                    search_q = q.removeprefix("search ").strip()
                                    if search_q and not q.startswith("update "):
                                        logger.warning(
                                            "[guardrail] nudge échoué → injection forcée search_catalog(%r)",
                                            search_q,
                                        )
                                        tool_calls = [{
                                            "function": {
                                                "name": "search_catalog",
                                                "arguments": {"query": search_q, "column": "nom_poste"},
                                            }
                                        }]
                                        llm_text = assistant_msg.get("content", "")
                                        if llm_text:
                                            messages.append({"role": "assistant", "content": llm_text})
                                        messages.append({
                                            "role": "assistant",
                                            "content": "",
                                            "tool_calls": tool_calls,
                                        })
                                    else:
                                        logger.warning("[guardrail] nudge échoué, tâche non injectable → break")
                                        break
                                else:
                                    break
                            else:
                                logger.warning(
                                    "[guardrail] nudge échoué, pas d'injection possible → break"
                                )
                                break
                    else:
                        # Pas de tâche en attente, pas d'action attendue → fin normale
                        break

                    # Reset nudge flag quand le LLM coopère (ou après injection)
                    self._guardrail_nudged = False
                    _synthetic_injection = True
                else:
                    _synthetic_injection = False

                # Append assistant tool-call message (skip si déjà ajouté par le guardrail)
                if not _synthetic_injection:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg.get("content", ""),
                            "tool_calls": tool_calls,
                        }
                    )

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name: str = fn.get("name", "")
                    tool_args = fn.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    logger.info("LLM tool call: %s args=%r", tool_name, tool_args)

                    # ask_user_choice: emit choice cards and stop immediately
                    if tool_name == "ask_user_choice":
                        # Utiliser les résultats catalog serveur si disponibles
                        # pour émettre des choice cards correctement typées (type "poste"/"affaire")
                        # avec le format JSON {nom_poste, occurrences} attendu par le frontend.
                        last_results = getattr(self, "_last_deduped_results", None)
                        choice_from_server = None
                        if last_results:
                            try:
                                _current_aff = await asyncio.to_thread(
                                    self.catalog.get_current_affaire, conversation_id
                                )
                                choice_from_server = _detect_conflict(last_results, _current_aff)
                            except Exception:
                                pass
                        if choice_from_server and choice_from_server.get("type") in ("poste", "affaire"):
                            yield {"choices": choice_from_server}
                        else:
                            # Fallback : options LLM sans type → simple boutons
                            question = tool_args.get("question", "Choisissez une option :")
                            options  = tool_args.get("options", [])
                            yield {"choices": {"question": question, "options": options}}
                        panier = await asyncio.to_thread(self.catalog.get_panier, conversation_id)
                        yield {"done": True, "panier": panier}
                        return

                    # report_findings: show component list and ask confirmation before catalog search
                    if tool_name == "report_findings":
                        components = tool_args.get("components", [])
                        context = tool_args.get("context", "")
                        catalog_matches = tool_args.get("catalog_matches") or []
                        doc_only_models = tool_args.get("doc_only_models") or []
                        suggestion = tool_args.get("suggestion", "")
                        yield {"tool_call": {"name": tool_name, "status": "running"}}
                        yield {"tool_call": {"name": tool_name, "status": "done"}}
                        # Build action buttons depending on context
                        if catalog_matches:
                            # Spec-aware flow: best match or search anyway
                            btn_label = suggestion or "Ajouter quand même"
                            options = [
                                {
                                    "id": json.dumps(
                                        {"action": "confirm_findings", "components": components},
                                        ensure_ascii=False,
                                    ),
                                    "label": btn_label,
                                    "detail": f"{len(components)} composant(s)",
                                },
                                {
                                    "id": json.dumps({"action": "cancel_findings"}, ensure_ascii=False),
                                    "label": "Annuler",
                                },
                            ]
                        else:
                            # Standard flow (docs → catalog)
                            options = [
                                {
                                    "id": json.dumps(
                                        {"action": "confirm_findings", "components": components},
                                        ensure_ascii=False,
                                    ),
                                    "label": "Chercher dans le catalogue",
                                    "detail": f"{len(components)} composant(s) à rechercher",
                                },
                                {
                                    "id": json.dumps({"action": "cancel_findings"}, ensure_ascii=False),
                                    "label": "Annuler",
                                },
                            ]
                        yield {
                            "choices": {
                                "type": "findings_confirmation",
                                "question": context or "Composants identifiés dans la documentation :",
                                "components": components,
                                "catalog_matches": catalog_matches,
                                "doc_only_models": doc_only_models,
                                "suggestion": suggestion,
                                "options": options,
                            }
                        }
                        panier = await asyncio.to_thread(self.catalog.get_panier, conversation_id)
                        yield {"done": True, "panier": panier}
                        return

                    yield {"tool_call": {"name": tool_name, "status": "running"}}
                    result = await self._execute_tool(
                        tool_name, tool_args, collection, conversation_id, catalog_method
                    )

                    # After search_catalog: server-side conflict/element detection
                    # (more reliable than relying on the LLM to call ask_user_choice)
                    if tool_name == "search_catalog":
                        try:
                            current_affaire = await asyncio.to_thread(
                                self.catalog.get_current_affaire, conversation_id
                            )
                            deduped_results = json.loads(result)
                            # Stocker pour réutilisation dans ask_user_choice
                            self._last_deduped_results = deduped_results

                            # Determine whether the query targeted postes or elements.
                            # Use catalog's Snowball tokenizer (handles plurals).
                            matches_postes = await asyncio.to_thread(
                                self.catalog.query_matches_postes,
                                tool_args.get("query", ""),
                                deduped_results,
                            )

                            query_str = tool_args.get("query", "")
                            column = tool_args.get("column", "nom_poste")

                            if column == "elements":
                                # User explicitly chose to search in elements (col H).
                                # _last_raw_results stashed by _execute_tool (unfiltered).
                                raw = getattr(self, "_last_raw_results", [])
                                choice_event = _detect_element_conflict(query_str, raw)
                                if choice_event is None and deduped_results:
                                    choice_event = _build_relevance_choice(
                                        query_str, deduped_results
                                    )

                            elif column != "nom_poste":
                                # ensemble / nom_affaire / fournisseur column search.
                                # Results already filtered to that column in _execute_tool.
                                # Show standard poste conflict cards (PRIORITY 1/2).
                                choice_event = _detect_conflict(deduped_results, current_affaire)

                            elif matches_postes:
                                # Default column="nom_poste" and nom_poste matches found.
                                # For multiple distinct postes: do NOT auto-emit choice cards.
                                # The LLM must decide: specs to verify → report_findings,
                                # no specs → ask_user_choice. The _next_action hint in the
                                # tool result enforces this.
                                conflict = _detect_conflict(deduped_results, current_affaire)
                                if conflict and conflict.get("type") == "affaire":
                                    # Single poste / multiple affaires: safe to auto-handle.
                                    choice_event = conflict
                                else:
                                    # Multiple distinct postes: emit a catalog_preview so the
                                    # user can see what was found while the LLM reasons.
                                    if len(deduped_results) > 1:
                                        yield {
                                            "catalog_preview": {
                                                "query": query_str,
                                                "total": len(deduped_results),
                                                "postes": [
                                                    {
                                                        "nom_poste": _s(r.get("nom_poste")),
                                                        "ensemble":  _s(r.get("ensemble")),
                                                        "nom_affaire": _s(r.get("nom_affaire")),
                                                    }
                                                    for r in deduped_results[:5]
                                                ],
                                            }
                                        }
                                    choice_event = None

                            else:
                                # Default column="nom_poste", no nom_poste match found.
                                # Propose the user to search in another column.
                                logger.info(
                                    "search_catalog: no nom_poste match for %r"
                                    " — emitting column choice card",
                                    query_str,
                                )
                                choice_event = _build_column_choice(query_str)

                        except Exception as exc:
                            logger.warning("conflict/element detection failed: %s", exc)
                            choice_event = None

                        # Enrich choice cards with table data (SQL — exact DB values)
                        if choice_event and choice_event.get("type") == "affaire":
                            nom_poste_ce = choice_event.get("nom_poste", "")
                            for opt in choice_event.get("options", []):
                                try:
                                    elements = await asyncio.to_thread(
                                        self.catalog.get_elements_for_poste,
                                        nom_poste_ce, opt["id"],
                                    )
                                    opt["columns"] = ["Ensemble", "Éléments", "Fournisseur", "Prix €"]
                                    opt["rows"] = [
                                        [
                                            el.get("ensemble") or "—",
                                            el.get("elements") or "—",
                                            el.get("fournisseur") or "—",
                                            str(el.get("fourniture") or "—"),
                                        ]
                                        for el in elements
                                    ]
                                    nb = len(elements)
                                    opt["description"] = f"{nb} élément{'s' if nb > 1 else ''}"
                                except Exception as exc:
                                    logger.warning("enrichment affaire failed: %s", exc)

                        elif choice_event and choice_event.get("type") == "poste":
                            poste_totals: dict[str, list] = {}
                            for row in deduped_results:
                                np = row.get("nom_poste", "")
                                if np:
                                    if np not in poste_totals:
                                        poste_totals[np] = []
                                    poste_totals[np].append((
                                        row.get("nom_affaire", ""),
                                        row.get("prix_total"),
                                    ))
                            for opt in choice_event.get("options", []):
                                try:
                                    parsed = json.loads(opt["id"])
                                    nom_poste_opt = parsed.get("nom_poste", "")
                                    affaires = poste_totals.get(nom_poste_opt, [])
                                    opt["columns"] = ["Affaire", "Prix total"]
                                    opt["rows"] = [
                                        [aff, f"{round(float(pt), 2)}€" if pt is not None else "—"]
                                        for aff, pt in affaires
                                    ]
                                    nb = len(affaires)
                                    opt["description"] = f"{nb} affaire{'s' if nb > 1 else ''}"
                                except Exception as exc:
                                    logger.warning("enrichment poste failed: %s", exc)

                        if choice_event:
                            yield {"tool_call": {"name": tool_name, "status": "done"}}
                            yield {"choices": choice_event}
                            panier = await asyncio.to_thread(
                                self.catalog.get_panier, conversation_id
                            )
                            yield {"done": True, "panier": panier}
                            return

                    # Track add_to_panier success for post-stream choice cards
                    if tool_name == "add_to_panier":
                        try:
                            _r = json.loads(result)
                            if _r.get("added", 0) > 0:
                                _add_to_panier_succeeded = True
                        except Exception:
                            pass

                    # After search_docs: emit a summary of what was found so the
                    # frontend can show the user which sources were consulted.
                    if tool_name == "search_docs":
                        try:
                            parsed = json.loads(result)
                            # result is {"chunks": [...], "catalog_postes": [...]}
                            inner_chunks = (
                                parsed.get("chunks", [])
                                if isinstance(parsed, dict)
                                else parsed
                            )
                            if inner_chunks:
                                sources = list({
                                    c.get("source", "").split("/")[-1]
                                    for c in inner_chunks
                                    if c.get("source")
                                })
                                yield {
                                    "docs_result": {
                                        "query": tool_args.get("query", ""),
                                        "sources": sources[:5],
                                        "count": len(inner_chunks),
                                    }
                                }
                            else:
                                yield {
                                    "docs_result": {
                                        "query": tool_args.get("query", ""),
                                        "sources": [],
                                        "count": 0,
                                    }
                                }
                        except Exception:
                            pass

                    # Emit settings update to frontend immediately
                    if tool_name == "set_devis_settings":
                        try:
                            parsed_settings = json.loads(result)
                            yield {"settings": {
                                "coefficient": parsed_settings.get("coefficient", 0.0),
                                "coef_final":  parsed_settings.get("coef_final", 0.0),
                            }}
                        except Exception:
                            pass

                    # Emit highlight event for visually marking the updated panier item
                    if tool_name == "update_panier_item":
                        try:
                            parsed_update = json.loads(result)
                            item_id = parsed_update.get("item_id")
                            if item_id:
                                yield {"highlight": item_id}
                        except Exception:
                            pass

                    messages.append({"role": "tool", "content": result})
                    yield {"tool_call": {"name": tool_name, "status": "done"}}

                # Emit updated panier after each tool round
                panier = await asyncio.to_thread(
                    self.catalog.get_panier, conversation_id
                )
                yield {"panier": panier}

                # If any update_panier_item was called this round, stop the loop.
                # The system-prompt panier context is stale (captured before the loop),
                # so the LLM would keep retrying the same update unnecessarily.
                if any(
                    tc.get("function", {}).get("name") == "update_panier_item"
                    for tc in tool_calls
                ):
                    break

            # ── Final streaming response (no tools passed) ────────────────────
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": True,
                        "think": False,  # Désactive le mode thinking qwen3
                    },
                ) as resp:
                    _in_think = False  # filtre tokens <think>...</think> (qwen3)
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                if "<think>" in token:
                                    _in_think = True
                                if _in_think:
                                    if "</think>" in token:
                                        _in_think = False
                                        after = token.split("</think>", 1)[1]
                                        if after:
                                            yield {"token": after}
                                else:
                                    yield {"token": token}
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as exc:
            logger.error("Devis chat error: %s", exc, exc_info=True)
            yield {"error": str(exc)}

        finally:
            panier = await asyncio.to_thread(self.catalog.get_panier, conversation_id)
            # Signal the frontend to intercept the next user message with a scope choice.
            # The choice cards are shown client-side before the next query is sent to the LLM.
            yield {"done": True, "panier": panier, "ask_scope": _add_to_panier_succeeded}

    async def generate_devis(
        self, conversation_id: str
    ) -> AsyncGenerator[dict, None]:
        """
        Stream the final devis markdown table from the confirmed panier items.
        Called when the user clicks "Générer le devis".
        """
        panier = await asyncio.to_thread(self.catalog.get_panier, conversation_id)

        if not panier:
            yield {
                "token": (
                    "Le panier est vide. Ajoutez des produits avant de générer le devis."
                )
            }
            yield {"done": True, "panier": []}
            return

        rows = "\n".join(
            f"- Poste: {p['nom_poste']} | Ensemble: {p.get('ensemble') or 'N/A'} | "
            f"Qté: {p.get('quantite', 1)} | "
            f"Fournisseur: {p.get('fournisseur') or 'N/A'} | "
            f"Fourniture: {p.get('fourniture') or 'N/A'} | "
            f"Affaire réf.: {p.get('nom_affaire') or 'N/A'}"
            for p in panier
        )

        prompt = (
            "Génère un devis structuré en tableau Markdown pour les postes ci-dessous.\n"
            "Colonnes exactes : Poste | Ensemble | Qté | Fournisseur | Fourniture | Affaire réf.\n"
            "Ajoute un récapitulatif et des notes si pertinent.\n\n"
            f"Postes confirmés :\n{rows}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant de génération de devis. "
                    "Génère uniquement le tableau Markdown demandé, sans commentaires superflus."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield {"token": token}
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.error("Generate devis error: %s", exc, exc_info=True)
            yield {"error": str(exc)}

        yield {"done": True, "panier": panier}
