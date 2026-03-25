"""
Orchestrator service — mode Agent.

Flux par requête
────────────────
1. classify_intent(message) via Ollama local → simple_claude | simple_rag | combined
2. Router vers l'agent approprié :
   - simple_claude : Claude API streaming (question générale, 0 données VLM)
   - simple_gpt    : OpenAI GPT streaming (question générale, 0 données VLM)
   - simple_rag    : RAGEngine local + Ollama streaming (données VLM uniquement)
   - combined      : Claude non-streaming → enrichit la query RAG → Ollama synthèse locale
   - combined_gpt  : GPT non-streaming → enrichit la query RAG → Ollama synthèse locale

Invariant confidentialité
─────────────────────────
Claude API / OpenAI API reçoivent : le message utilisateur uniquement.
Ces APIs ne voient JAMAIS : chunks VLM, noms clients, specs propriétaires.
La synthèse finale (combined / combined_gpt) est faite par Ollama en local.

SSE event shapes
────────────────
{"agent_step": "classifying"}                            — classification en cours
{"agent_step": "claude_query"}                           — appel Claude API
{"agent_step": "gpt_query"}                              — appel OpenAI API
{"agent_step": "rag_search"}                             — recherche docs locaux
{"agent_step": "synthesizing"}                           — synthèse Ollama
{"agent": "simple_claude|simple_gpt|simple_rag|combined|combined_gpt"}
{"token": "..."}                                         — token de réponse
{"sources": [...]}                                       — sources RAG si applicable
{"done": true}                                           — fin de stream
{"error": "..."}                                         — erreur
"""

import asyncio
import json
import logging
import os
import threading
from collections.abc import AsyncGenerator
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
ENABLE_NO_THINK = os.environ.get("ENABLE_NO_THINK", "false").lower() == "true"

IntentType = Literal["simple_claude", "simple_gpt", "simple_rag", "combined", "combined_gpt"]

_CLASSIFICATION_SYSTEM = """\
Tu es un classificateur d'intentions. Analyse la question et réponds UNIQUEMENT \
par l'un de ces 3 mots exacts :

- simple_claude  : question générale ne nécessitant pas de données internes VLM \
(veille technologique, marché, stratégie, rédaction, calcul, connaissance générale)
- simple_rag     : question spécifique aux données internes VLM \
(produits VLM, catalogue, specs techniques, projets VLM, machines VLM, clients VLM)
- combined       : question nécessitant à la fois des connaissances générales \
ET des données internes VLM

Réponds UNIQUEMENT par le mot exact, rien d'autre."""

_CLAUDE_SYSTEM = """\
Tu es un assistant expert en veille technologique, budgétaire et stratégique \
pour l'industrie manufacturière et robotique avancée. \
Réponds avec précision et concision en français."""

_SYNTHESIS_SYSTEM = """\
Tu es un assistant expert de VLM Robotics. Synthétise une réponse complète \
en combinant des informations générales et des données spécifiques VLM Robotics. \
Structure clairement les deux parties. Réponds en français."""


def _no_think(prompt: str) -> str:
    """Préfixe /no_think pour Qwen3 si ENABLE_NO_THINK est actif."""
    return "/no_think\n\n" + prompt if ENABLE_NO_THINK else prompt


class OrchestratorService:

    def __init__(
        self,
        api_key: str,
        claude_model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
        openai_api_key: str = "",
        openai_model: str = "gpt-4o",
    ) -> None:
        self._api_key = api_key
        self._claude_model = claude_model
        self._max_tokens = max_tokens
        self._openai_api_key = openai_api_key
        self._openai_model = openai_model
        self._claude_adapter = None  # lazy init
        self._openai_adapter = None  # lazy init

    def _get_claude_adapter(self):
        if self._claude_adapter is None:
            from backend.adapters.claude_api_adapter import ClaudeApiAdapter
            self._claude_adapter = ClaudeApiAdapter(
                api_key=self._api_key,
                model=self._claude_model,
                max_tokens=self._max_tokens,
            )
        return self._claude_adapter

    def _get_openai_adapter(self):
        if self._openai_adapter is None:
            from backend.adapters.openai_adapter import OpenAIAdapter
            self._openai_adapter = OpenAIAdapter(
                api_key=self._openai_api_key,
                model=self._openai_model,
                max_tokens=self._max_tokens,
            )
        return self._openai_adapter

    # ─── Intent classification ────────────────────────────────────────────────

    async def classify_intent(self, message: str) -> IntentType:
        """Classify message intent using local Ollama (non-streaming, fast)."""
        prompt = _no_think(message)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": _CLASSIFICATION_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 20},
                    },
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"].strip().lower()
                # Strip potential <think>...</think> if model ignores /no_think
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                if "simple_claude" in content:
                    return "simple_claude"
                if "simple_rag" in content:
                    return "simple_rag"
                if "combined" in content:
                    return "combined"
                logger.warning("Classification imprécise: '%s' → fallback combined", content)
                return "combined"
        except Exception as exc:
            logger.error("Erreur classification intent: %s → fallback combined", exc)
            return "combined"

    # ─── Main entrypoint ─────────────────────────────────────────────────────

    async def stream_response(
        self,
        message: str,
        collection_name: str,
        force_mode: IntentType | None = None,
    ) -> AsyncGenerator[str, None]:
        """Classify then route — main SSE generator."""
        if force_mode:
            intent: IntentType = force_mode
            yield f"data: {json.dumps({'agent': intent, 'agent_step': 'forced'})}\n\n"
        else:
            yield f"data: {json.dumps({'agent_step': 'classifying'})}\n\n"
            intent = await self.classify_intent(message)
            yield f"data: {json.dumps({'agent': intent})}\n\n"

        if intent == "simple_claude":
            async for chunk in self._stream_claude(message):
                yield chunk
        elif intent == "simple_gpt":
            async for chunk in self._stream_gpt(message):
                yield chunk
        elif intent == "simple_rag":
            async for chunk in self._stream_rag(message, collection_name):
                yield chunk
        elif intent == "combined_gpt":
            async for chunk in self._stream_combined_with_adapter(
                message, collection_name, provider="gpt"
            ):
                yield chunk
        else:
            async for chunk in self._stream_combined_with_adapter(
                message, collection_name, provider="claude"
            ):
                yield chunk

    # ─── simple_claude ───────────────────────────────────────────────────────

    async def _stream_claude(self, message: str) -> AsyncGenerator[str, None]:
        """Stream Claude API response — no VLM data involved."""
        if not self._api_key:
            yield f"data: {json.dumps({'error': 'ANTHROPIC_API_KEY non configuré'})}\n\n"
            return

        yield f"data: {json.dumps({'agent_step': 'claude_query'})}\n\n"
        try:
            adapter = self._get_claude_adapter()
            async for text in adapter.stream(message, system=_CLAUDE_SYSTEM):
                yield f"data: {json.dumps({'token': text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            logger.error("Claude streaming error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    # ─── simple_gpt ──────────────────────────────────────────────────────────

    async def _stream_gpt(self, message: str) -> AsyncGenerator[str, None]:
        """Stream OpenAI GPT response — no VLM data involved."""
        if not self._openai_api_key:
            yield f"data: {json.dumps({'error': 'OPENAI_API_KEY non configuré'})}\n\n"
            return

        yield f"data: {json.dumps({'agent_step': 'gpt_query'})}\n\n"
        try:
            adapter = self._get_openai_adapter()
            async for text in adapter.stream(message, system=_CLAUDE_SYSTEM):
                yield f"data: {json.dumps({'token': text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            logger.error("GPT streaming error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    # ─── simple_rag ──────────────────────────────────────────────────────────

    async def _stream_rag(self, message: str, collection_name: str) -> AsyncGenerator[str, None]:
        """RAG + local Ollama stream (thread+queue pattern, same as chat.py)."""
        from backend.api.dependencies import get_collection_manager
        from backend.core.search import RAGEngine

        yield f"data: {json.dumps({'agent_step': 'rag_search'})}\n\n"

        cm = get_collection_manager()
        if not cm.collection_existe(collection_name):
            yield f"data: {json.dumps({'error': f'Collection {collection_name!r} introuvable'})}\n\n"
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run() -> None:
            try:
                rag = RAGEngine(collection_name, collection_manager=cm)
                result = rag.generer_avec_sources(message, stream=True)
                for token in result["reponse"]:
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", result["sources"]))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        threading.Thread(target=_run, daemon=True).start()

        while True:
            try:
                kind, data = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'error': 'Timeout — aucune réponse du pipeline RAG'})}\n\n"
                break
            if kind == "token":
                yield f"data: {json.dumps({'token': data})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'sources': data, 'done': True})}\n\n"
                break
            elif kind == "error":
                yield f"data: {json.dumps({'error': data})}\n\n"
                break

    # ─── combined / combined_gpt ─────────────────────────────────────────────

    async def _stream_combined_with_adapter(
        self,
        message: str,
        collection_name: str,
        provider: str = "claude",  # "claude" | "gpt"
    ) -> AsyncGenerator[str, None]:
        """External LLM general context → enrich RAG query → Ollama synthesis (local).

        provider="claude" → uses ClaudeApiAdapter
        provider="gpt"    → uses OpenAIAdapter
        """
        from backend.api.dependencies import get_collection_manager
        from backend.core.search import RAGEngine

        # Select adapter and check key
        if provider == "gpt":
            if not self._openai_api_key:
                yield f"data: {json.dumps({'error': 'OPENAI_API_KEY non configuré'})}\n\n"
                return
            step_event = "gpt_query"
            adapter = self._get_openai_adapter()
            fallback_label = "OpenAI GPT"
        else:
            if not self._api_key:
                async for chunk in self._stream_rag(message, collection_name):
                    yield chunk
                return
            step_event = "claude_query"
            adapter = self._get_claude_adapter()
            fallback_label = "Claude API"

        # Step 1 — External LLM non-streaming (only the user message, no VLM data)
        yield f"data: {json.dumps({'agent_step': step_event})}\n\n"
        external_context = ""
        try:
            external_context = await adapter.query(message, system=_CLAUDE_SYSTEM)
        except Exception as exc:
            logger.warning("%s error in combined mode: %s — falling back to RAG", fallback_label, exc)
            async for chunk in self._stream_rag(message, collection_name):
                yield chunk
            return

        # Step 2 — RAG search with enriched query
        yield f"data: {json.dumps({'agent_step': 'rag_search'})}\n\n"
        enriched_query = f"{message}\n\nContexte général pertinent :\n{external_context[:600]}"

        cm = get_collection_manager()
        if not cm.collection_existe(collection_name):
            yield f"data: {json.dumps({'error': f'Collection {collection_name!r} introuvable'})}\n\n"
            return

        loop = asyncio.get_running_loop()
        rag_queue: asyncio.Queue = asyncio.Queue()

        def _run_rag() -> None:
            try:
                rag = RAGEngine(collection_name, collection_manager=cm)
                ctx_text, sources = rag.rechercher(enriched_query)
                loop.call_soon_threadsafe(rag_queue.put_nowait, ("done", (ctx_text, sources)))
            except Exception as exc:
                loop.call_soon_threadsafe(rag_queue.put_nowait, ("error", str(exc)))

        threading.Thread(target=_run_rag, daemon=True).start()
        try:
            kind, rag_data = await asyncio.wait_for(rag_queue.get(), timeout=120.0)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': 'Timeout — recherche RAG trop longue'})}\n\n"
            return

        if kind == "error":
            yield f"data: {json.dumps({'error': rag_data})}\n\n"
            return

        rag_context, sources = rag_data

        # Step 3 — Ollama local synthesis (streaming)
        yield f"data: {json.dumps({'agent_step': 'synthesizing'})}\n\n"

        vlm_section = rag_context[:3000] if rag_context else "Aucun document interne trouvé sur ce sujet."
        synthesis_prompt = _no_think(
            f"L'utilisateur a demandé : {message}\n\n"
            f"--- Contexte général (connaissances expertes) ---\n{external_context}\n\n"
            f"--- Informations spécifiques VLM Robotics (base documentaire interne) ---\n{vlm_section}\n\n"
            "Synthétise une réponse complète et structurée en français. "
            "Distingue clairement ce qui relève de la connaissance générale "
            "et ce qui est spécifique à VLM Robotics."
        )

        synth_queue: asyncio.Queue = asyncio.Queue()

        def _run_synthesis() -> None:
            import requests as _req
            try:
                r = _req.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": _SYNTHESIS_SYSTEM},
                            {"role": "user", "content": synthesis_prompt},
                        ],
                        "stream": True,
                        "options": {"temperature": 0.3},
                    },
                    stream=True,
                    timeout=120,
                )
                for line in r.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                loop.call_soon_threadsafe(synth_queue.put_nowait, ("token", token))
                            if data.get("done"):
                                loop.call_soon_threadsafe(synth_queue.put_nowait, ("done", None))
                                break
                        except json.JSONDecodeError:
                            pass
            except Exception as exc:
                loop.call_soon_threadsafe(synth_queue.put_nowait, ("error", str(exc)))

        threading.Thread(target=_run_synthesis, daemon=True).start()

        while True:
            try:
                kind, data = await asyncio.wait_for(synth_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'error': 'Timeout — synthèse trop longue'})}\n\n"
                break
            if kind == "token":
                yield f"data: {json.dumps({'token': data})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"
                break
            elif kind == "error":
                yield f"data: {json.dumps({'error': data})}\n\n"
                break
