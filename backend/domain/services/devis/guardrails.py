"""
Guardrails server-side : détection des situations où le LLM aurait dû appeler un tool.
Extrait de devis_service.py pour être testable unitairement.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Sentinel retourné par detect_expected_tool quand un nudge est nécessaire
# mais qu'aucun tool synthétique ne peut être construit.
NEEDS_NUDGE = {"_needs_nudge": True}


def detect_multi_action(message: str) -> list[str] | None:
    """
    Détecte les demandes multi-actions dans le message utilisateur.
    Retourne une liste d'items pour plan_tasks, ou None si mono-action.
    """
    msg = message.strip()

    # Skip messages système et sélections utilisateur
    if msg.startswith("[SYSTÈME]") or msg.startswith("Sélectionné") or msg.startswith("Cherche"):
        return None
    if len(msg) < 10:
        return None

    action_verbs = re.findall(
        r"^(ajoute|cherche|recherche|trouve|mets|modifie|update|supprime|passe)\b",
        msg.lower(),
    )
    action_prefix = action_verbs[0] if action_verbs else ""

    msg_lower = msg.lower()
    work_text = msg_lower
    if action_prefix:
        work_text = re.sub(r"^" + re.escape(action_prefix) + r"\s+", "", work_text)

    parts = re.split(r"\s+et\s+|,\s*", work_text)
    parts = [p.strip() for p in parts if p.strip()]

    cleaned = []
    for p in parts:
        p = re.sub(r"^(moi|toi|lui|nous|vous|eux|une|un|les|le|la|des|d'|l[''])\s*", "", p).strip()
        p = re.sub(r"^(une|un|les|le|la|des)\s+", "", p).strip()
        if p and len(p) >= 2:
            cleaned.append(p)

    if len(cleaned) < 2:
        return None

    items = []
    for item_text in cleaned:
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


def detect_expected_tool(
    messages: list[dict],
    task_list: list[dict] | None,
) -> dict | None:
    """
    Analyse l'état courant pour déterminer si le LLM aurait dû appeler un tool
    au lieu de générer du texte.

    Retourne :
    - Un dict {"name": "...", "arguments": {...}} → tool call synthétique à exécuter
    - NEEDS_NUDGE → le LLM doit réessayer avec un rappel
    - None → fin normale (pas de tool attendu)
    """
    # ── Cas 3 : search_docs sans report_findings ──────────────────────────
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
                    logger.info(
                        "[guardrail] _next_action présent mais LLM a généré du texte → nudge"
                    )
                    return NEEDS_NUDGE
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Cas 1 : tâches en attente ─────────────────────────────────────────
    if task_list:
        pending = [t for t in task_list if not t.get("done")]
        if pending:
            next_query = pending[0].get("query", "")
            q_lower = next_query.lower().strip()
            if q_lower.startswith("update "):
                logger.info(
                    "[guardrail] tâche update en attente %r → nudge (pas d'injection)",
                    next_query,
                )
                return NEEDS_NUDGE
            else:
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

    return None
