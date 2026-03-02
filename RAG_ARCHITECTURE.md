# Architecture RAG — chatbot-local

Document technique complet décrivant la pipeline RAG implémentée dans ce projet.
Objectif : audit externe et identification d'axes d'amélioration.

---

## Contexte

Chatbot local pour interroger une base documentaire d'entreprise (offres commerciales,
dossiers techniques, contrats). Documents majoritairement en **français**, format PDF/DOCX/Excel.
Contrainte matérielle : **RTX 3060 Laptop (6 Go VRAM)** — tous les modèles doivent cohabiter.

Stack : FastAPI + Next.js + ChromaDB + Ollama + Docker Compose.

---

## Vue d'ensemble des deux pipelines

```
INGESTION
fichier → Parser → Chunker → Embedder (CPU) → ChromaDB

RETRIEVAL + GÉNÉRATION
question → [Query Rewriting] → Embedder → Vector Search ─┐
                                                           ├→ RRF Fusion → Reranker → LLM → SSE stream
                             BM25 Search ─────────────────┘
                             [Keyword Fallback si score < 0.01]
```

---

## Pipeline 1 : Ingestion

### Étape 1 — Parsing

**Librairies :** Docling (prioritaire), pymupdf4llm (fallback PDF), python-docx, pandas/xlrd/openpyxl

| Format | Parser | Notes |
|--------|--------|-------|
| PDF | Docling | Layout AI + extraction de tableaux (TableFormer). Mode `accurate` activé. |
| DOCX | Docling | Meilleure extraction que python-docx sur les tableaux |
| TXT / MD | lecture directe | — |
| CSV | pandas | — |
| XLSX / XLS | Docling (XLSX) / pandas row-grouping (XLS) | Fallback `xlrd` si Docling échoue |

**Docling config (docker-compose) :**
- `USE_DOCLING=true`
- `DOCLING_OCR=false` (OCR désactivé — lourd, ~1.5 GB de modèles supplémentaires)
- `DOCLING_TABLE_MODE=accurate` (TableFormer — meilleure extraction de tableaux, +lent)

**Output :** liste de `ParsedPage(texte: str, source: str, page: int)`

---

### Étape 2 — Extraction de métadonnées

Le nom de fichier est parsé par regex pour extraire :
- `machine` : GEMINI / SOLO / COMPAQT / HYMANCO (détecté dans le nom)
- `type_doc` : devis / offre / dossier_technique / rfi
- `ref_projet` : AP0xxx ou suite de 4-6 chiffres
- `client` : extrait du pattern `"NNNN - CLIENT - ..."`

Ces métadonnées sont stockées dans ChromaDB avec chaque chunk.
Elles servent ensuite au **filtrage metadata** lors du retrieval.

---

### Étape 3 — Chunking

**Deux modes disponibles (configurable par env var) :**

#### Mode RecursiveCharacterTextSplitter (défaut si semantic chunking échoue)
- Librairie : `langchain-text-splitters`
- Chunk size : **450 tokens** (sweet spot benchmarks RAG sur docs techniques)
- Overlap : **67 tokens** (~15%)
- Séparateurs : `["\n\n", "\n", ". ", " ", ""]`
- Tokenizer : `AutoTokenizer` de `mixedbread-ai/mxbai-embed-large-v1` (même tokenizer que l'embedding model)

#### Mode SemanticChunker (activé via `USE_SEMANTIC_CHUNKING=true`)
- Librairie : `langchain-experimental`
- Principe : découpe aux **frontières sémantiques** (changement de sujet détecté par embeddings)
- Seuil : percentile 95 (coupe seulement aux ruptures très nettes)
- Fallback automatique sur RecursiveCharacterTextSplitter si SemanticChunker indisponible
- **Actuellement activé** (`USE_SEMANTIC_CHUNKING=true` dans docker-compose)

**Protection anti-overflow :** les chunks > 490 tokens sont re-splitté avant indexation.

---

### Étape 3b — Contextual Retrieval (optionnel, désactivé par défaut)

- Technique : [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- Principe : chaque chunk est enrichi d'un contexte généré par le LLM
  (ex: "Ce chunk concerne les conditions de garantie de la cellule GEMINI pour le projet SAFRAN")
- Le chunk indexé devient : `[contexte LLM]\n\n[chunk original]`
- Réduit les erreurs de retrieval de ~67% selon Anthropic
- **Désactivé** (`USE_CONTEXTUAL_RETRIEVAL=false`) — nécessite N appels Ollama à l'ingest,
  trop lent sur 6 Go VRAM avec llama3.1:8b
- Config : `CONTEXTUAL_MAX_WORKERS=3` (parallélisme des appels Ollama)

---

### Étape 4 — Embedding & Indexation

**Modèle :** `mixedbread-ai/mxbai-embed-large-v1`
- Dimension : 1024
- Fenêtre contexte : 512 tokens
- Spécialisé retrieval (pré-entraîné avec instruction prefixes)
- Prefix doc : `""` (vide pour mxbai-embed-large)
- Prefix query : `"Represent this sentence for searching relevant passages: "`

**Mode d'exécution : HuggingFace CPU** (`USE_HF_EMBEDDINGS=true`)
- Librairie : `sentence-transformers` via `SentenceTransformer` directement
- Device : **CPU** — libère la VRAM entière pour le LLM
- Raison : sur 6 Go VRAM, charger le modèle d'embedding sur GPU évictait le LLM
  à chaque requête (swap de ~30s). CPU = ~100-200ms de latence, imperceptible.
- Normalisation : `normalize_embeddings=True` (nécessaire pour cosine similarity avec ChromaDB)
- Singleton via `@lru_cache(maxsize=1)` — le modèle (~700 Mo) est chargé une seule fois

**Vector store :** ChromaDB
- Service dédié (Docker), persist sur volume
- Interface LangChain : `langchain-chroma`
- Tracking incrémental : SHA256 des fichiers stocké dans `metadata.json` par collection
  (évite de ré-indexer un document non modifié)

---

## Pipeline 2 : Retrieval + Génération

### Étape 1 — Paramètres dynamiques

`k` et `num_ctx` sont calculés en fonction de la taille réelle de la collection :

```
k       = max(6, min(nb_chunks // 8, 20))
num_ctx = k × 250 tokens/chunk + 1200 overhead, borné [4096, 32768]
```

Exemples : 48 chunks → k=6, num_ctx=4096 | 446 chunks → k=20, num_ctx=8192

---

### Étape 2 — Query Rewriting (si historique de conversation)

- Librairie : appel direct Ollama `/api/generate`
- Modèle : llama3.1:8b
- Activé uniquement à partir du **2e tour de conversation** (le 1er n'a pas d'historique utile)
- Principe : reformule la question en requête autonome riche en mots-clés
  (gère les références anaphoriques : "cette machine", "le projet en question"...)
- Exemple : `"comment est donc implantée la cellule en question"` →
  `"implantation cellule DED métallique SAFRAN VLM Robotics"`
- Timeout : 15s — en cas d'échec, la question originale est utilisée

---

### Étape 3 — Pré-filtrage metadata

Si la question contient un nom de machine connu (GEMINI, SOLO, COMPAQT, HYMANCO),
un filtre ChromaDB `{machine: {$eq: "GEMINI"}}` est appliqué à la recherche vectorielle.
En cas d'échec avec filtre, retry sans filtre.

---

### Étape 4 — Recherche vectorielle

- `k_candidats = min(k × 3, 25)` — on récupère 3× plus de candidats pour le reranking
- `similarity_search_with_score(question, k=k_candidats)`
- La question est embedée avec le **même modèle et le même prefix** que les documents
- ChromaDB retourne les k_candidats chunks les plus proches en distance cosine

---

### Étape 5 — Hybrid Search : BM25 + RRF Fusion

**BM25 (`USE_HYBRID_SEARCH=true`) :**
- Librairie : `rank-bm25` (BM25Okapi)
- Tokenization : `re.findall(r'\w+')` + stemmer Snowball français (NLTK)
- Le stemmer normalise singulier/pluriel : "directives" et "directive" → même racine
- Index construit lazily depuis ChromaDB, **invalidé automatiquement** si la collection change
- Cherche k_candidats dans le BM25

**RRF Fusion (Reciprocal Rank Fusion) :**
- Combine vector results + BM25 results
- Score RRF = Σ 1/(60 + rang) pour chaque liste (60 = constante standard de la littérature)
- Avantage : agnostique aux échelles de scores — fusionne des scores incomparables
- Output : k_candidats meilleurs chunks fusionnés avec scores RRF (~0.016–0.032)

**Note :** les scores RRF sont des scores de rang relatif, **pas** des scores de similarité absolue.

---

### Étape 6 — Reranking

**Reranker BGE (`USE_RERANKER=true`) :**
- Librairie : `FlagEmbedding` >= 1.3.0
- Modèle : `BAAI/bge-reranker-v2-m3` (~570 Mo, multilingue FR/EN/ZH)
- Type : **cross-encoder** — lit la paire (question, chunk) ensemble, plus précis que bi-encoder
- `compute_score(pairs, normalize=True, max_length=512)` → sigmoid(logit) ∈ [0, 1]
- Tri des k_candidats par score décroissant, conservation des top k
- Singleton lazy chargé au premier appel, fp16 si GPU dispo

**Alternative ColBERT (`USE_COLBERT=false`) :**
- Librairie : `ragatouille`
- Modèle : `colbert-ir/colbertv2.0` (~2 Go)
- Type : late-interaction (un vecteur par token, scoring MaxSim token-à-token)
- Meilleur sur acronymes et termes techniques courts
- **Désactivé** — modèle trop lourd pour cohabiter avec llama3.1:8b sur 6 Go VRAM

**Observation connue :** les scores BGE pour des requêtes conversationnelles courtes
(ex: "parle moi de X") sont naturellement bas (~0.01–0.02). Le reranker est mieux calibré
pour des questions factorielles directes ("conditions de paiement projet X"). La sélection
relative reste cependant correcte (le chunk le plus pertinent a toujours le score le plus haut).

---

### Étape 7 — Keyword Fallback

- Déclenché si `score_best < KEYWORD_FALLBACK_THRESHOLD` (0.01)
- Recherche exacte via `ChromaDB.where_document($contains: mot)` sur les mots significatifs
  de la question (stopwords filtrés)
- Variantes morphologiques tentées : singulier/pluriel, différentes casses
- Score attribué : **0.5** (valeur arbitraire, pas une similarité réelle)
- Sert de filet de sécurité uniquement — si vraiment aucun résultat sémantique n'est trouvé

---

### Étape 8 — Post-processing

**Seuil relatif reranker :**
- Élimine les chunks dont le score < score_max × 0.1
- Le meilleur chunk passe toujours

**Seuil absolu :**
- Élimine les chunks avec score < 0.01
- Évite d'injecter du contexte non pertinent dans le prompt

**Déduplication PDF/DOCX :**
- Si un même document existe en deux formats (offre.pdf + offre.docx),
  le chunk avec le score le plus bas est éliminé (par couple source_base + page)

---

### Étape 9 — Génération (LLM + SSE Streaming)

**Modèle :** `llama3.1:8b` via Ollama
- Quantisation : Q4_K_Medium (4.58 Go)
- GPU : 32/33 layers offloadées sur RTX 3060 (3.99 Go VRAM)
- 1 layer sur CPU (411 Mo RAM)
- KV cache : 512 Mo VRAM (4096 tokens de contexte)
- Flash Attention : activé automatiquement

**Prompts :**
- `defaut` : prompt générique (langue neutre)
- `vlm_robotics` : prompt spécialisé avec persona commercial VLM Robotics, consignes de citation
- Sélection automatique par nom de collection (collection "vlm" → prompt vlm_robotics)
- Prompts custom chargeables depuis `prompts.json`

**Mémoire de conversation :**
- Les 10 derniers messages (User/Assistant) sont passés dans le prompt
- Utilisés aussi pour le Query Rewriting (reformulation avec contexte conversationnel)

**Streaming :**
- Protocole : SSE (Server-Sent Events) via FastAPI + `sse-starlette`
- `stream=True` dans la requête Ollama → tokens streamés au fil de la génération
- Keep-alive : `"keep_alive": "30m"` (maintient le LLM en VRAM 30 min après la dernière requête)

---

## Paramètres clés (docker-compose.yml)

| Variable | Valeur | Rôle |
|----------|--------|------|
| `USE_HF_EMBEDDINGS` | true | Embeddings sur CPU (libère VRAM) |
| `EMBED_HF_MODEL` | mixedbread-ai/mxbai-embed-large-v1 | Modèle d'embedding |
| `OLLAMA_EMBED_MODEL` | mxbai-embed-large | Nom Ollama (si USE_HF_EMBEDDINGS=false) |
| `USE_DOCLING` | true | Parser avancé PDF/DOCX |
| `DOCLING_TABLE_MODE` | accurate | TableFormer pour extraction tableaux |
| `USE_RERANKER` | true | Cross-encoder BGE |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | Modèle reranker |
| `USE_HYBRID_SEARCH` | true | BM25 + vector RRF |
| `USE_SEMANTIC_CHUNKING` | true | Découpage aux frontières sémantiques |
| `SEMANTIC_BREAKPOINT_THRESHOLD` | 95 | Percentile seuil de rupture |
| `USE_CONTEXTUAL_RETRIEVAL` | false | Enrichissement chunks par LLM (désactivé) |
| `USE_COLBERT` | false | Reranker ColBERT (désactivé, trop lourd) |
| `KEYWORD_FALLBACK_THRESHOLD` | 0.01 | Seuil déclenchement fallback keyword |

---

## Contraintes matérielles et compromis

**VRAM 6 Go = contrainte principale :**

| Composant | Mémoire | Device |
|-----------|---------|--------|
| llama3.1:8b (Q4_K_M) | ~3.99 Go VRAM + 411 Mo RAM | GPU (32/33 layers) |
| KV cache (4096 ctx) | 512 Mo VRAM | GPU |
| Compute graph | ~669 Mo VRAM | GPU |
| mxbai-embed-large-v1 | ~700 Mo RAM | **CPU** (choix délibéré) |
| BGE reranker v2-m3 | ~570 Mo VRAM/RAM | GPU si dispo |
| Docling (TableFormer) | ~600 Mo RAM | CPU |
| BM25 index | quelques Mo RAM | CPU |

**Choix CPU pour les embeddings :**
Mettre le modèle d'embedding sur GPU déclenchait son éviction par le LLM à chaque requête
(swap de ~30s, inacceptable en production). Sur CPU, latence ~100-200ms, imperceptible.

---

## Limitations et points ouverts

1. **Scores reranker bas pour requêtes conversationnelles**
   BGE v2-m3 calibré pour requêtes factuelles. Scores ~0.01-0.02 pour "parle moi de X"
   vs 0.4-0.8 pour "conditions de paiement projet X". La sélection relative reste correcte.

2. **Troncature embeddings à 350 chars**
   `_MAX_CHARS = 350` (~250 tokens) pour rester sous la limite de 512 tokens du modèle.
   Les chunks peuvent être tronqués si trop longs (tableaux Docling ~1.4 char/token).

3. **Contextual Retrieval désactivé**
   Technique la plus impactante sur la qualité du retrieval (-67% erreurs), mais incompatible
   avec les contraintes VRAM actuelles sans augmentation du temps d'ingest.

4. **Fenêtre contexte LLM limitée à 4096 tokens**
   Contrainte OLLAMA_CONTEXT_LENGTH=4096. Avec k=6 chunks × ~250 tokens = 1500 tokens de
   contexte maximum, plus overhead prompt (~1200). Peu de marge pour les grandes collections.

5. **Pas de multi-collection search**
   La recherche est faite sur une seule collection à la fois. Pas de federated search.

6. **Query Rewriting uniquement sur l'historique**
   La reformulation de requête n'est pas appliquée à la première question d'une conversation.

7. **BM25 en mémoire (non persisté)**
   L'index BM25 est reconstruit à chaque redémarrage du backend (mais rapide).
