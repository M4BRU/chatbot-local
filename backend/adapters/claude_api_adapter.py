"""Claude API adapter — wrapper autour du SDK Anthropic.

Fournit deux opérations :
- query()  : appel non-streaming, retourne le texte complet (combined mode step 1)
- stream() : appel streaming, itère les tokens (simple_claude mode)

Contrainte absolue : ne jamais transmettre de données internes VLM à ce client.
L'appelant (OrchestratorService) est responsable de cette séparation.
"""

import logging
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class ClaudeApiAdapter:
    """Thin async wrapper around the Anthropic Python SDK."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6", max_tokens: int = 2048) -> None:
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def query(self, message: str, system: str | None = None) -> str:
        """Non-streaming call — returns full text response."""
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": message}],
        }
        if system:
            kwargs["system"] = system
        resp = await self._client.messages.create(**kwargs)
        return resp.content[0].text

    async def stream(self, message: str, system: str | None = None) -> AsyncIterator[str]:
        """Streaming call — yields text tokens as they arrive."""
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": message}],
        }
        if system:
            kwargs["system"] = system
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
