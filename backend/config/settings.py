"""Application settings."""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Ollama settings
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3.5:4b"
    # ollama_embed_model inutilisé si USE_HF_EMBEDDINGS=true (bge-m3 via HuggingFace CPU)
    ollama_embed_model: str = "mxbai-embed-large"

    # HuggingFace embeddings (actif si USE_HF_EMBEDDINGS=true)
    embed_hf_model: str = "BAAI/bge-m3"
    use_hf_embeddings: bool = True

    # ChromaDB settings (fallback — actif si VECTOR_DB=chroma)
    chroma_host: str = "chromadb"
    chroma_port: int = 8100

    # Qdrant settings (actif si VECTOR_DB=qdrant)
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    # Document storage
    documents_path: str = "/app/documents"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://frontend:3000"]

    # Claude API (mode agent orchestrateur)
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    agent_max_tokens: int = 2048

    # OpenAI API (modes simple_gpt / combined_gpt)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Auth & database
    database_url: str = "postgresql+asyncpg://chatbot:chatbot@postgres:5432/chatbot"
    jwt_secret_key: str = "changeme-generate-with-openssl-rand-hex-32"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 30
    cookie_secure: bool = False
    cookie_samesite: str = "lax"
    rbac_config_path: str = "/app/config/rbac.yaml"

    # Whisper transcription service
    whisper_url: str = "http://whisper:9000"
    whisper_timeout: int = 600

    # LangGraph feature flag (USE_LANGGRAPH=true active la nouvelle architecture)
    use_langgraph: bool = False

    # Langfuse observability (100% local, self-hosted)
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://langfuse:3000"
