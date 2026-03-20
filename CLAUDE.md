# CLAUDE.md — Instructions pour tous les agents IA (BMAD Dev, Architect, etc.)

## RÈGLE CRITIQUE — Lire avant d'implémenter

**Avant toute implémentation (Epic, Story, Quick Spec), l'agent DOIT consulter :**

1. `docs/technical-research-index.md` — index de toutes les recherches techniques récentes
2. `_bmad-output/planning-artifacts/research/` — fichiers de recherche détaillés

Ces fichiers contiennent des décisions techniques récentes (2026) qui **priment sur toute connaissance antérieure** du modèle sur les technologies concernées (Whisper, Qwen, llama.cpp, multi-agents, etc.).

---

## Migration Hardware Prévue — RTX 5090 (32GB VRAM)

**Le projet va migrer vers une nouvelle machine :**

| Composant | Spec cible |
|---|---|
| GPU | RTX 5090 (32GB VRAM) |
| CPU | AMD Ryzen 9 9900X (12c/24t, Zen5) |
| RAM | 64GB DDR5 |
| SSD | 2TB NVMe PCIe Gen4 |

**Implications pour l'implémentation :**

- Ne PAS sur-optimiser pour 6GB VRAM — ce sera obsolète. Concevoir pour 32GB.
- Les configs "current 6GB" sont des hacks temporaires (qwen3.5:4b actif, pas de Whisper simultané)
- Sur 5090 : qwen3.5:9b confortable, Whisper large-v3 + LLM simultanément, vLLM viable
- Les abstractions hexagonales (ports/adapters) doivent rendre la migration transparente

**Ce qui change avec 5090 vs config actuelle (6GB) :**

| Aspect | Actuel (6GB) | 5090 (32GB) |
|---|---|---|
| LLM cible | qwen3.5:4b (3.4GB) | qwen3.5:9b ou Qwen3:14B+ |
| Whisper | Container séparé obligatoire | Peut tourner en parallèle du LLM |
| Inference server | Ollama | Ollama ou vLLM (pour throughput) |
| Multi-modèles | 1 à la fois | Plusieurs simultanément |
| Contextual retrieval | qwen3:0.6b (0.6GB) | Peut utiliser un modèle plus grand |

---

## Stack technique actuelle

- Backend : FastAPI (hexagonal architecture, ports/adapters)
- Frontend : Next.js 14 (App Router, TypeScript)
- LLM : Ollama (**qwen3.5:4b** — modèle actif confirmé 2026-03-19)
- Embeddings : mxbai-embed-large (HuggingFace CPU)
- Vector DB : ChromaDB (migration Qdrant prévue Phase 2)
- Docker Compose avec GPU passthrough (CUDA)
- Language : français (commentaires code), English (documents techniques)

---

## Fichiers clés à connaître

| Fichier | Contenu |
|---|---|
| `backend/domain/services/devis_service.py` | Boucle tool-calling + system prompt devis |
| `backend/adapters/catalog_adapter.py` | SQLite catalog, panier, task list |
| `backend/core/document_manager.py` | Ingestion PDF + contextual retrieval |
| `backend/config/settings.py` | Variables d'env (OLLAMA_MODEL, etc.) |
| `docker-compose.yml` | Services Docker |
| `.env` | Configuration locale (ne pas committer) |
