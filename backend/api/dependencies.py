"""Dependency injection wiring for FastAPI."""

from functools import lru_cache

from backend.config.settings import Settings


@lru_cache
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()


# Port implementations will be registered here as adapters are implemented
# Example:
# def get_llm_port() -> LlmPort:
#     settings = get_settings()
#     return OllamaAdapter(settings.ollama_url)
