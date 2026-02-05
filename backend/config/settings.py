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
    ollama_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

    # ChromaDB settings
    chroma_host: str = "chromadb"
    chroma_port: int = 8100

    # Document storage
    documents_path: str = "/app/documents"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://frontend:3000"]
