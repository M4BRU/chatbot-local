# syntax=docker/dockerfile:1
# Backend Dockerfile - Multi-stage build
#
# Deux modes :
#   Standard : FROM python:3.12-slim AS builder (rebuild ML si nécessaire)
#   Rapide   : décommenter FROM chatbot-local-ml-base AS builder après avoir lancé
#              docker build -f backend/Dockerfile.base -t chatbot-local-ml-base .
#
FROM python:3.12-slim AS builder
# FROM chatbot-local-ml-base AS builder  # ← décommenter après build base

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Étape 1 : dépendances ML lourdes (Docling + FlagEmbedding + torch).
# Couche séparée — cachée tant que requirements-ml.txt ne change pas.
# --mount=type=cache : le cache pip est persisté sur le host entre les builds.
# Même si la couche Docker est invalidée, les packages déjà téléchargés
# ne sont jamais re-téléchargés (gain majeur sur torch ~2GB, docling, etc.)
COPY backend/requirements-ml.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements-ml.txt

# Étape 2 : dépendances légères (FastAPI, LangChain, parsers…).
# Invalidée seulement si requirements.txt change, sans retoucher les ML.
COPY backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# Production stage
FROM python:3.12-slim AS production

WORKDIR /app

# System libs required by Docling (PDF rendering via libGL/X11)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY backend/ ./backend/

# Create documents directory
RUN mkdir -p /app/documents

# backend/core/ doit prendre la priorité sur core/ racine pour les imports
# PYTHONPATH=/app/backend:/app → "import core" résout backend/core/ en premier
ENV PYTHONPATH=/app/backend:/app

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
