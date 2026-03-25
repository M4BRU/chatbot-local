"""
DevisState — LangGraph state definition for the devis mode.
Shared between hard and soft harness graphs.
"""

import operator
from typing import Annotated, Literal

from typing_extensions import TypedDict


class DevisState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────────
    conversation_id: str
    collection: str
    catalog_method: str              # "bm25" | "sql"
    harness_mode: Literal["hard", "soft"]

    # ── User input ────────────────────────────────────────────────────────────
    user_message: str
    history: list[dict]              # [{role, content}]

    # ── Phase 0: Analyse documentaire (lazy, au premier lancement harness) ───
    doc_analysis_cached: bool        # True si l'analyse existe déjà en cache
    doc_structure: dict | None       # Structure extraite (sections, TOC)
    doc_requirements: list[dict]     # Exigences extraites [{id, text, type, section}]
    doc_tables: list[dict]           # Tableaux extraits comme unités
    doc_summary: str | None          # Résumé global du document (500 mots)

    # ── RFQ Planner ───────────────────────────────────────────────────────────
    is_rfq: bool
    dimensions: list[dict]           # [{dimension, query}]
    # Annotated with operator.add → append reducer (nodes can add findings without replacing all)
    findings: Annotated[list[dict], operator.add]
    gaps: list[dict]                 # Gaps détectés
    rfq_context: str | None
    rfq_candidates: list[dict]

    # ── Tool-calling loop ─────────────────────────────────────────────────────
    messages: list[dict]             # Format Ollama [{role, content, tool_calls?}]
    tool_iteration: int
    guardrail_nudged: bool
    add_to_panier_succeeded: bool
    should_stop: bool

    # ── Panier / catalog (lu depuis adapter) ─────────────────────────────────
    panier: list[dict]
    task_list: list[dict]
    devis_settings: dict

    # ── Phase tracking ────────────────────────────────────────────────────────
    current_phase: str | None
    completed_phases: list[str]

    # ── Soft harness ──────────────────────────────────────────────────────────
    todos: list[dict]                # [{id, label, done}]


def initial_state(
    conversation_id: str,
    collection: str,
    user_message: str,
    history: list[dict],
    catalog_method: str = "bm25",
    harness_mode: Literal["hard", "soft"] = "hard",
) -> DevisState:
    """Create the initial state for a new devis turn."""
    return DevisState(
        conversation_id=conversation_id,
        collection=collection,
        catalog_method=catalog_method,
        harness_mode=harness_mode,
        user_message=user_message,
        history=history,
        doc_analysis_cached=False,
        doc_structure=None,
        doc_requirements=[],
        doc_tables=[],
        doc_summary=None,
        is_rfq=False,
        dimensions=[],
        findings=[],
        gaps=[],
        rfq_context=None,
        rfq_candidates=[],
        messages=[],
        tool_iteration=0,
        guardrail_nudged=False,
        add_to_panier_succeeded=False,
        should_stop=False,
        panier=[],
        task_list=[],
        devis_settings={},
        current_phase=None,
        completed_phases=[],
        todos=[],
    )
