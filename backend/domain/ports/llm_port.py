"""LLM Port - Interface for language model operations."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LlmPort(ABC):
    """Port interface for LLM inference operations."""

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Generate streaming response from LLM.

        Args:
            prompt: User prompt with context
            system_prompt: Optional system prompt override
            model: Optional model name override

        Yields:
            Token chunks as they are generated
        """
        pass

    @abstractmethod
    async def check_health(self) -> dict:
        """Check LLM service health.

        Returns:
            Dict with status and details
        """
        pass

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names
        """
        pass
