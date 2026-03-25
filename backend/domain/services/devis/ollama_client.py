"""Shared Ollama HTTP client — reused by all nodes and tools."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

_HTTP_CLIENT: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Retourne un AsyncClient partagé (connection pooling, pas de leak TCP)."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=600.0)
    return _HTTP_CLIENT


async def ollama_chat(messages: list[dict], tools: list | None = None) -> dict:
    """Non-streaming Ollama /api/chat call. Used for tool-calling loop and single LLM calls."""
    payload: dict = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        # think=True : le LLM raisonne avant chaque décision d'outil.
        # Le bloc <think> va dans response["message"]["thinking"] (champ séparé d'Ollama),
        # PAS dans "content" — donc il n'est PAS ajouté au message history.
        # Zéro pollution du contexte, meilleure qualité de décision.
        "think": True,
        "options": {"num_ctx": 8192},
    }
    if tools:
        payload["tools"] = tools

    client = _get_http_client()
    resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    resp.raise_for_status()
    data = resp.json()

    thinking = data.get("message", {}).get("thinking", "")
    if thinking:
        logger.info(
            "[ollama_chat] 🧠 thinking (%d chars):\n%s",
            len(thinking), thinking[:1000],
        )
    return data


async def ollama_chat_structured(
    messages: list[dict],
    schema: dict,
    think: bool = False,
    temperature: float = 0,
    num_ctx: int = 8192,
) -> dict:
    """Non-streaming Ollama call with grammar-constrained JSON output (format=schema)."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": think,
        "format": schema,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    client = _get_http_client()
    resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    resp.raise_for_status()
    return resp.json()


async def ollama_chat_simple(
    messages: list[dict],
    think: bool = False,
    temperature: float = 0,
    num_ctx: int = 8192,
) -> dict:
    """Non-streaming Ollama call without tools or schema (plain text output)."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    client = _get_http_client()
    resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    resp.raise_for_status()
    return resp.json()
