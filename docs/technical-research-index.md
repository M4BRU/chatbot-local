# Index des Recherches Techniques — chatbot-local

_Mis à jour : 2026-03-10_

> **Pour les agents BMAD** : consulter les fichiers listés ci-dessous AVANT d'implémenter toute story technique. Ces recherches sont récentes et contiennent des décisions qui priment sur les connaissances générales.

---

## Fichiers de recherche disponibles

### 1. Qwen3.5 — Évaluation LLM (2026-03-06)
**Fichier** : `_bmad-output/planning-artifacts/research/technical-Qwen3.5b-research-2026-03-06.md`

**Concerne** : Stories 7.x (Devis), toute story impliquant le LLM local

**Décisions clés** :
- Remplacer `qwen3:8b` par `qwen3.5:4b` (3.4GB, 256K context, meilleur tool-calling)
- Sur 5090 : utiliser `qwen3.5:9b` à la place
- `qwen3.5:4b` disponible via `ollama pull qwen3.5:4b`
- ik_llama.cpp : 5x plus rapide que mainline llama.cpp sur CPU pour Qwen3.5 (si migration vers llama.cpp un jour)
- vLLM bat SGLang pour modèles AWQ MoE Qwen3 (pour le futur sur 5090)
- Méthodologie benchmark GPU : framework → context max → MCR optimal

---

### 2. Whisper & Transcription (2026-03-06)
**Fichier** : `_bmad-output/planning-artifacts/research/technical-whisper-transcription-2026-03-06.md`

**Concerne** : Epic 8 (Transcription), Stories 8.1, 8.2

**Décisions clés** :
- **Silero VAD obligatoire** en pre-gate avant tout appel Whisper (~90% hallucinations évitées)
- `condition_on_previous_text=False` — critique pour stopper les cascades
- Blocklist FR à constituer (inspiré de [Vexa-ai](https://github.com/Vexa-ai/vexa))
- Backend recommandé : `faster-whisper large-v3-turbo` (container Docker séparé)
- Parakeet éliminé : anglais uniquement
- Sur 5090 : Whisper large-v3 + LLM peuvent tourner simultanément (plus besoin de container séparé)
- Voxtral (Mistral) à surveiller pour FR

**Comparatif modèles ASR 2026** :
| Modèle | WER | FR | CPU | Notes |
|---|---|---|---|---|
| Whisper large-v3-turbo | ~6% | ✅ | Oui | Notre choix |
| Parakeet TDT v3 | ~6% | ❌ | Oui | Anglais only |
| Canary-Qwen 2.5B | 5.63% | ? | Non | GPU requis |
| Voxtral | ? | Probable | ? | À évaluer |

---

### 3. Architecture Multi-Agents — Pièges à éviter (2026-03-06)
**Source** : Reddit r/LocalLLaMA — retour production (agents "arguing")

**Concerne** : toute story impliquant plusieurs agents LLM

**Règles à appliquer** :
- Agents **stateless** — pas de contexte partagé entre tâches
- Orchestrateur = **code Python déterministe**, pas un LLM qui décide du routing
- `temperature=0` pour toute logique de délégation/routing
- L'orchestrateur envoie les prompts en tant que `"user"` — les agents ne savent pas qu'ils parlent à un autre agent
- Notre architecture actuelle (boucle Python + LLM sans état) est déjà le bon pattern
- Paper RUMAD (arxiv 2602.23864) : à lire avant d'implémenter un vrai système multi-agents

---

### 4. Infrastructure de serving (2026-03-06)
**Source** : Reddit r/LocalLLaMA — llama-swap, ik_llama.cpp

**Concerne** : architecture infra, migration llama.cpp

**Options documentées** :

| Outil | Usage | Pertinence actuelle | Sur 5090 |
|---|---|---|---|
| **Ollama** | Serving LLM simple | ✅ Actuel | ✅ Toujours valide |
| **llama-swap** | Proxy multi-modèles multi-engines | 🔮 Phase 2+ | Intéressant |
| **ik_llama.cpp** | llama.cpp fork 5x plus rapide CPU Qwen3.5 | 📋 Si migration | Moins pertinent (GPU) |
| **vLLM** | High-throughput serving | 🔮 Phase 3+ | ✅ Sur 5090 |
| **llama.cpp router** | Multi-modèles sans proxy | 🔮 Alternative llama-swap | Possible |

**Avantage llama-swap** : router les requêtes vers différents modèles (ex: `qwen3.5:4b` pour devis, `qwen3:8b` pour RAG) sans changer le code applicatif. Via config YAML + feature `filters` (forcer temperature par modèle).

---

## Roadmap Hardware — Impact sur les décisions d'implémentation

### Config actuelle (Phase 1, 6GB VRAM)
- Contraintes VRAM sévères → choix conservateurs (qwen3.5:4b, Whisper container séparé)
- Ne PAS créer de dépendances à la contrainte 6GB dans le code métier

### Config cible (Phase 2, RTX 5090 32GB)
- LLM : qwen3.5:9b ou Qwen3:14B
- Whisper : large-v3-turbo en parallèle du LLM
- Inference : Ollama ou vLLM
- Qdrant à la place de ChromaDB (voir `future-evolution-ai-agents-hardware.md`)

### Modèles cibles par config hardware

| Config | LLM recommandé | Taille VRAM |
|---|---|---|
| Actuel 6GB | qwen3.5:4b | 3.4 GB |
| **RTX 5090 32GB** | **Qwen3.5-35B-A3B (MoE)** | **~22 GB** |
| RTX 5090 32GB (alt) | Qwen3.5-27B (dense) | ~18 GB |
| RTX 5090 32GB (léger) | Qwen3.5-9B | ~7 GB |

> Qwen3.5-35B-A3B = 34.66B params mais ~3B actifs/forward (MoE) → rapide malgré la taille. Candidat principal sur 5090.
> Qwen3-Coder = modèle coding uniquement → **non pertinent** pour notre usage (tool-calling FR + RAG).

### Règle d'implémentation
> Toujours abstraire le nom de modèle et la config d'inférence derrière des variables d'env (`OLLAMA_MODEL`, `WHISPER_MODEL`, etc.). Ne jamais hardcoder `qwen3.5:4b` dans la logique métier.

---

## 5. RAG Pipeline — Findings critiques (Reddit r/LocalLLaMA)

### 5a. Hierarchical/Tree-based RAG (RAPTOR) — NE PAS IMPLÉMENTER

**Benchmark sérieux (paper : https://doi.org/10.5281/zenodo.18714001)** :

| Approche | nDCG@10 (22k chunks) |
|---|---|
| Flat dense + cross-encoder reranker | **0.749** |
| RAPTOR tree-based routing | **0.094** ← inutilisable |

**Cause** : les erreurs de routing se multiplient à chaque niveau d'arbre. À 15% de miss/niveau, après 5 niveaux < 50% de queries correctement routées. Plus le corpus est grand, pire c'est.

**→ Notre approche BM25 + dense hybrid est déjà le bon pattern.** Ne jamais implémenter RAPTOR/tree routing.

**Amélioration concrète à prévoir** : ajouter un **cross-encoder reranker** après la récupération hybride. Stack qui fonctionne en prod :
```
BM25 + dense embeddings → RRF fusion → cross-encoder reranker → enrichissement nœuds parents
```
- Source: [Reddit r/LocalLLaMA — Hierarchical RAG benchmarks](https://www.reddit.com/r/LocalLLaMA/)

### 5a-bis. Parent/Child chunking — À NE PAS CONFONDRE avec RAPTOR

**Parent/child = enrichissement de contexte ✅** (différent du routing RAPTOR ❌)

```
[Chunk parent = paragraphe entier]
    ├── [Chunk child 1 = phrase A]
    ├── [Chunk child 2 = phrase B]  ← matche la query
    └── [Chunk child 3 = phrase C]
```

On indexe les **petits chunks** (précis, meilleure similarité vectorielle), mais à la récupération on remonte chercher le **chunk parent** (contexte complet) à passer au LLM. Pas de navigation d'arbre → pas d'erreurs en cascade.

**Gain concret** : meilleure précision de retrieval (petits chunks) + meilleure qualité de réponse LLM (contexte large). C'est l'un des éléments du stack gagnant du benchmark RAPTOR.

**Référence implémentation** : [github.com/GiovanniPasq/agentic-rag-for-dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies) — pipeline complet : PDF → Markdown → parent/child chunking → hybrid retrieval → Qdrant → reranker → Ollama.

**À prévoir pour Phase 2** (upgrade Qdrant) : implémenter parent/child chunking dans le pipeline d'ingestion.

---

### 5b. Scaling des embeddings — SentenceTransformer à grande échelle

**Contexte** : SentenceTransformer CPU + FastAPI → goulot d'étranglement sous charge concurrente (asyncio ne parallélise pas le CPU).

**Notre situation actuelle** : 500 messages/jour non-concurrent → pas de problème. Notre pattern `asyncio.to_thread()` wrappant les appels HF CPU est correct.

**À faire sur 5090 (Phase 2)** : passer les embeddings sur GPU via vLLM :
```bash
vllm serve mxbai-embed-large --task embed
```
Gain : ~10x vs CPU, 1M+ embeddings/heure sur T4 (source communauté). Élimine complètement le goulot.

- Source: [Reddit r/LocalLLaMA — Scaling embeddings vLLM](https://www.reddit.com/r/LocalLLaMA/)
- Docs: [vLLM Embedding Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

### 5c. Semantic caching RAG — Pattern Phase 2

Pour éviter de recomputer embeddings + LLM sur des questions répétées :

```python
cache_key = hash(query + context_hash + model + temperature + system_prompt)
# Même query + même corpus = même réponse instantanée, 0 token consommé
```

Pas besoin d'outil dédié (AvocadoDB est trop early-stage). Implémentable avec Redis ou SQLite + hash SHA-256 des chunks récupérés.

**Pertinent à partir de Phase 2** quand le volume de requêtes augmente.

---

### 6. BM25 Field Weighting + Qwen3.5 Quantization (2026-03-09)
**Fichier** : `_bmad-output/planning-artifacts/research/technical-bm25-qwen35-quantization-research-2026-03-09.md`

**Sources** : 2 posts Reddit r/LocalLLaMA + 8 recherches web

**Concerne** : Migration Qdrant (Group B), Story 7.3, hardware migration 5090

**Décisions clés** :
- **Re-indexation complète obligatoire** pour ChromaDB→Qdrant (issue #5106 — vecteurs incompatibles)
- **BM25 field weighting** possible via dual sparse vectors (bm25-title + bm25-body) — pas de BM25F natif Qdrant
- **`indexing_threshold=0`** pendant le bulk upload Qdrant → réactiver après (performance)
- **Qwen3.5:9b sorti le 02/03/2026** — bat gpt-oss-120B, tool calling TAU2-Bench 79.1 — modèle 5090 uniquement (6GB insuffisant)
- **Qwen3.5-35B-A3B** : 194 tok/s sur RTX 5090, 22GB int4 — candidat principal confirmé
- **Ollama streaming tool calling** : disponible mais toujours 2 phases (accumulation + réponse) — notre plan Story 7.3 (non-streaming tools + streaming final) reste le bon
- **Stack RAG prod 2026** : hybrid recall 0.72→0.91, précision 0.68→0.87 vs BM25 seul ; +15–25% sur corpus domain-specific

**Hardware actuel : RTX 3060 Laptop 6GB**
- qwen3.5:4b ✅ | qwen3.5:9b ❌ (KV cache insuffisant) | qwen3.5:35b-A3B ❌

---

### 7. Excel RAG Optimization — Stratégies d'organisation (2026-03-10)
**Fichier** : `_bmad-output/planning-artifacts/research/technical-excel-rag-optimization-2026-03-10.md`

**Sources** : Reddit posts (RAG on Excel files) + 5 recherches web + audit code `parsers.py` + `catalog_adapter.py`

**Concerne** : pipeline d'ingestion Excel, `parsers.py`, `document_manager.py`, futur upload de fichiers structurés

**Décisions clés** :
- **Deux pipelines distincts = correct** : SQL+BM25 (catalogue) vs RAG vectoriel (docs génériques) — ne pas mélanger
- **Quick wins Phase A** (sans rebuild) : metadata enrichies (sheet_name, row_range, column_names), `_df_to_prose()` avec unités, semantic block chunking
- **Routing automatique** à implémenter : heuristique sur noms de colonnes → catalogue → `catalog_adapter.py` ; sinon → RAG vectoriel
- **Row-as-Document** : meilleur pour tableaux larges riches ; block-chunking sémantique > bloc de 30 lignes fixes
- **Multi-sheet** : une collection/document par feuille + metadata cross-sheet pour les relations
- **LlamaParse / pgvector** : over-engineering non justifié pour le volume actuel
- **Sur RTX 5090** : `bge-m3` embeddings + Hybrid RRF (vector + BM25) + cross-encoder reranker pour les résultats tabulaires

**Plan d'implémentation** :
- Phase A (Quick Wins, ~2-3h, sans rebuild) : metadata + `_df_to_prose()` amélioré + semantic grouping
- Phase B (Routing, 1 jour, rebuild) : `detect_excel_type()` + auto-routing upload
- Phase C (Sur RTX 5090) : `bge-m3` + Hybrid RRF + cross-encoder reranker tabulaire
