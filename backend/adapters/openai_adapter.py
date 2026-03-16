"""OpenAI API adapter — wrapper autour du SDK openai.

Fournit deux opérations :
- query()  : appel non-streaming, retourne le texte complet (combined_gpt step 1)
- stream() : appel streaming, itère les tokens (simple_gpt mode)

Même interface que ClaudeApiAdapter — interchangeable dans l'orchestrateur.

Contrainte absolue : ne jamais transmettre de données internes VLM à ce client.
L'appelant (OrchestratorService) est responsable de cette séparation.
"""

import logging
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """Thin async wrapper around the OpenAI Python SDK."""

    def __init__(self, api_key: str, model: str = "gpt-4o", max_tokens: int = 2048) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def query(self, message: str, system: str | None = None) -> str:
        """Non-streaming call — returns full text response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        resp = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    async def stream(self, message: str, system: str | None = None) -> AsyncIterator[str]:
        """Streaming call — yields text tokens as they arrive."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        stream = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
