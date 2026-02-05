"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.dependencies import get_settings
from backend.api.routes import health_router

settings = get_settings()

app = FastAPI(
    title="chatbot-local",
    description="RAG chatbot for VLM Robotics product knowledge",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)


@app.get("/")
async def root():
    """Root endpoint - basic JSON response."""
    return {"message": "chatbot-local API", "version": "0.1.0"}
