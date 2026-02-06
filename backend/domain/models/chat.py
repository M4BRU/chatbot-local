"""Chat domain models."""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="User message")
    collection_name: str = Field(..., min_length=1, description="ChromaDB collection to search")
    prompt_name: str = Field(default="defaut", description="Prompt template name")
    history: list[ChatMessage] = Field(default=[], description="Previous messages for context")


class ChatSource(BaseModel):
    """Source document reference."""

    fichier: str
    page: str | int
    score: float


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    response: str
    sources: list[ChatSource]
