"""User tools: ask_user_choice, set_devis_settings, write_todos, read_todos."""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


async def execute_ask_user_choice(tool_args: dict) -> str:
    # ask_user_choice is handled as a SSE event — its result is emitted to frontend,
    # not returned as a tool result. Return a confirmation string.
    question = tool_args.get("question", "")
    options = tool_args.get("options", [])
    logger.info("ask_user_choice: question=%r, %d options", question[:80], len(options))
    return json.dumps({"presented": True, "question": question, "options_count": len(options)})


async def execute_set_devis_settings(
    tool_args: dict,
    conversation_id: str,
    catalog,
) -> str:
    coefficient = float(tool_args.get("coefficient", -1))
    coef_final  = float(tool_args.get("coef_final", -1))
    current = await asyncio.to_thread(catalog.get_devis_settings, conversation_id)
    if coefficient < 0:
        coefficient = current.get("coefficient", 0.0)
    if coef_final < 0:
        coef_final = current.get("coef_final", 0.0)
    await asyncio.to_thread(catalog.set_devis_settings, conversation_id, coefficient, coef_final)
    return json.dumps({"coefficient": coefficient, "coef_final": coef_final})


async def execute_write_todos(tool_args: dict, state_updater) -> str:
    """Soft harness only — updates the todos list in state via state_updater callback."""
    todos = tool_args.get("todos", [])
    # Normalize: ensure each todo has required fields
    normalized = [
        {"id": t.get("id", str(i)), "label": t.get("label", ""), "done": t.get("done", False)}
        for i, t in enumerate(todos)
        if isinstance(t, dict)
    ]
    if state_updater:
        await state_updater(normalized)
    logger.info("write_todos: %d todos", len(normalized))
    return json.dumps({"todos": normalized})


async def execute_read_todos(current_todos: list[dict]) -> str:
    """Soft harness only — reads the current todos from state."""
    return json.dumps({"todos": current_todos})
