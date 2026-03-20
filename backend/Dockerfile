# syntax=docker/dockerfile:1
# Backend Dockerfile — code-only rebuild (~10 sec).
#
# Pré-requis : avoir buildé l'image de base ML une fois :
#   docker build -f backend/Dockerfile.base -t chatbot-local-ml-base .
#
# Rebuild normal (changement de code) :
#   docker compose up -d --build backend   → ~10 sec
#
# Si un fichier requirements-*.txt change :
#   docker build -f backend/Dockerfile.base -t chatbot-local-ml-base .
#   docker compose up -d --build backend

FROM python:3.12-slim

WORKDIR /app

# Runtime system libs (Docling PDF rendering via libGL/X11)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances ML lourdes viennent de l'image de base (pré-installées)
COPY --from=chatbot-local-ml-base /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Dépendances légères (auth, SQL, etc.) — installées ici pour éviter rebuild de la base
COPY backend/requirements-infra.txt ./backend/requirements-infra.txt
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --user -r backend/requirements-infra.txt -r backend/requirements.txt --quiet

# Code applicatif uniquement
COPY backend/ ./backend/

# Répertoire documents persistant
RUN mkdir -p /app/documents

# backend/core/ prioritaire sur core/ racine pour les imports
ENV PYTHONPATH=/app/backend:/app

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--timeout-keep-alive", "0"]
