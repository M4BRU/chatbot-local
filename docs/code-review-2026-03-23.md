# Code Review Complet — 2026-03-23

Scope : **13 500 lignes** sur 22 fichiers backend + frontend.
Reviewer : Claude Opus 4.6 (5 agents paralleles)

---

## 1. Issues Critiques (a fixer en priorite)

### CRIT-1: 6 endpoints sans authentification — `collections.py`
- **Fichier:** `backend/api/routes/collections.py` lignes 46, 57, 74, 113, 209, 220
- `create_collection`, `get_collection`, `delete_collection`, `list_sources`, `list_chunks`, `version_status` n'ont aucun `Depends(get_current_user)`.
- Seul `list_collections` (ligne 31) est protege.
- **Fix:** Ajouter `Depends(_admin_role)` sur create/delete, `Depends(get_current_user)` sur les autres.

### CRIT-2: SQL Injection via NL2SQL — `excel_collection_adapter.py`
- **Fichier:** `backend/adapters/excel_collection_adapter.py:420-430`
- `_execute_sql` execute le SQL genere par le LLM avec seulement un check `startswith("SELECT")`.
- Un LLM manipule peut generer : `SELECT * FROM sqlite_master`, `SELECT load_extension(...)`, sous-requetes sur tables systeme.
- **Fix:** Ouvrir en read-only (`?mode=ro`), utiliser `conn.set_authorizer()`, whitelist tables `xl_*`, ajouter `LIMIT 200` force.

### CRIT-3: CRAG dedup ne fonctionne jamais — `search.py`
- **Fichier:** `backend/core/search.py:1516`
- `r not in fallback_crag` compare par identite objet (Document n'a pas `__eq__`), toujours `True`.
- **Fix:** Dedup par `doc.page_content[:150]` hash.

### CRIT-4: Mauvais CollectionManager dans `chat_sync` — `chat.py`
- **Fichier:** `backend/api/routes/chat.py:257`
- Cree `CollectionManager()` directement au lieu de `get_collection_manager()` (singleton Qdrant).
- **Fix:** Remplacer par `get_collection_manager()`.

### CRIT-5: Route collision `/{name}` capture `version-status` — `collections.py`
- **Fichier:** `backend/api/routes/collections.py:209`
- `GET /version-status` est defini apres `GET /{name}` — jamais atteint.
- **Fix:** Deplacer la route `version-status` avant `/{name}`.

### CRIT-6: `search_docs` parse `chunks` au lieu de `components` — `devis_service.py`
- **Fichier:** `backend/domain/services/devis_service.py:2789-2803`
- `parsed.get("chunks", [])` mais `_execute_tool` retourne `{"components": [...], "sources": [...]}`.
- Resultat toujours vide (count: 0, sources: []).
- **Fix:** Utiliser `parsed.get("components", [])` et `parsed.get("sources", [])`.

### CRIT-7: `max(pages)` crash si `None` dans le set — `search.py`
- **Fichier:** `backend/core/search.py:715`
- `pages` peut contenir `None` (de `metadata.get("page")`). `max(pages) - min(pages)` leve `TypeError`.
- **Fix:** Filtrer `numeric_pages = {p for p in pages if isinstance(p, (int, float))}`.

---

## 2. Issues High (bugs, securite, perf)

| # | Fichier | Issue | Fix |
|---|---|---|---|
| HIGH-1 | `chat.py:68` | Erreurs persistence avalees (`except: pass`) + pas de logger | Ajouter logger + `logger.exception(...)` |
| HIGH-2 | `devis_service.py:2951` | `generate_devis` ne filtre pas `<think>` tags | Ajouter `"think": False` ou filtre |
| HIGH-3 | `document_manager.py:444` | `_validate_title_with_ai` retourne `True` en erreur | Retourner `False` (conservateur) |
| HIGH-4 | `excel_collection_adapter.py:427` | Pas de LIMIT sur SQL LLM → OOM possible | `fetchmany(200)` ou injecter LIMIT |
| HIGH-5 | `orchestrator_service.py:257` | Pas de timeout sur `queue.get()` → block infini | `asyncio.wait_for(..., timeout=N)` |

---

## 3. Issues Medium (qualite, perf, duplication)

| # | Fichier | Issue |
|---|---|---|
| MED-1 | `devis_service.py` (x8) | Nouveau `httpx.AsyncClient` par appel LLM — extraire `_ollama_call()` + client partage |
| MED-2 | `devis_service.py:2395` | Panier/tasks re-fetch chaque iteration (24 DB calls inutiles/boucle) |
| MED-3 | `search.py:560` | MMR re-embed des chunks deja embedes (centaines de ms) |
| MED-4 | `devis/page.tsx:1744` | `setMessages(.map())` O(n) sur chaque token SSE |
| MED-5 | `devis.py:169` + `documents.py:206` | `_synthesize()` dupliquee quasi-identique dans 2 routes |
| MED-6 | `catalog_adapter.py:36-58` | BM25/stemmer reimplemente (existe deja dans `bm25_utils.py`) |
| MED-7 | Routes variees | Import paths `core.*` vs `backend.core.*` inconsistants |
| MED-8 | `catalog_adapter.py:61` vs `excel_collection_adapter.py:64` | `_norm()` defini differemment |
| MED-9 | 3 fichiers | Ollama HTTP calls directs au lieu de passer par un port |
| MED-10 | `dependencies.py` + `main.py` | Double `init_db()` |
| MED-11 | `search.py:901` | `_keyword_fallback_search` = 18 queries sequentielles |
| MED-12 | `devis_service.py:106,424,454` | `import re as _re` inside function body (deja importe) |
| MED-13 | `excel_collection_adapter.py:214` | Row-by-row INSERT (lent pour gros Excel) |

---

## 4. Frontend — Issues React

| # | Fichier | Issue | Fix |
|---|---|---|---|
| FE-1 | `page.tsx` | God component 810L, 25+ useState | Extraire hooks |
| FE-2 | `page.tsx:1534` | `handleSend` 170L, 10+ event types | Hook `useDevisChat` |
| FE-3 | `page.tsx:111` | Stale closure `SearchWorkspaceModal` useEffect | Ref pour `ws` |
| FE-4 | `page.tsx:1780` | `handleSend` recree chaque keystroke (dep `input`) | Lire input via ref |
| FE-5 | `page.tsx:96-112` | Mock code en prod (VLM-2024-001 hardcode) | Supprimer |
| FE-6 | `page.tsx:2050` | `MessageItem` pas memoize | `React.memo` |
| FE-7 | `page.tsx:2118` | `DevisPanel` pas memoize | `React.memo` |
| FE-8 | `page.tsx:1471` | Side-effect dans setState updater | `useEffect` debounce |

---

## 5. Plan de Refactoring

### Backend: `devis_service.py` (2974 L) → `backend/domain/services/devis/`

| Module | Contenu | ~Lignes |
|---|---|---|
| `prompts.py` | System prompt, tool definitions, JSON schemas | 170 |
| `conflict_detection.py` | Element/column/relevance/affaire conflict | 400 |
| `rfq_planner.py` | Decompose, search, gap detect, synthesize | 700 |
| `guardrails.py` | Multi-action detection, expected tool nudge | 200 |
| `context_compression.py` | Estimate chars, compress tool results | 100 |
| `devis_service.py` | Orchestrateur (execute_tool, chat_stream, generate) | 800 |

### Backend: `search.py` (2087 L) → `backend/core/search/`

| Module | Contenu | ~Lignes |
|---|---|---|
| `config.py` | Env vars, constants, context brackets | 170 |
| `prompts.py` | PROMPT_DEFAUT, PROMPT_VLM, domain context | 90 |
| `filters.py` | Image filter, source cap, PDF/DOCX dedup | 150 |
| `bm25.py` | BM25 index, cache, stemmer | 100 |
| `fusion.py` | RRF fusion, MMR | 70 |
| `reranker.py` | Reranker + ColBERT singletons | 100 |
| `query_transform.py` | Multi-query, HyDE, query rewrite | 180 |
| `retrieval_augment.py` | Tail extension, section, keyword fallback | 300 |
| `engine.py` | RAGEngine (rechercher decompose en steps) | 400 |
| `reasoning.py` | ReasoningEngine | 200 |

### Frontend: `devis/page.tsx` (2132 L)

**Hooks a extraire :**

| Hook | State |
|---|---|
| `useDevisChat` | messages, isLoading, error, activeToolCalls, rfqPlanning, candidates |
| `useDevisPanier` | postes, devisSettings, highlightedId, isExporting |
| `useAutoScroll` | showScrollBtn, scrollAreaRef |
| `useStreamingMessage` | streamingContent, streamingId (evite O(n) map/token) |
| `useLLMStatus` | llmReady, collections, catalogLoaded |

**Composants a extraire :** `DevisInputBar`, `WelcomeScreen`, `ChatView`, `MessageList`, `DevisHeader`

### Duplications cross-fichiers a eliminer

| Duplique | Source → Cible |
|---|---|
| BM25/stemmer | `catalog_adapter.py` → importer `bm25_utils.py` |
| `_norm()` | 2 fichiers → shared util |
| `_synthesize()` | 2 routes → service |
| Ollama HTTP | 3 fichiers → `OllamaPort` |
| `_resolve_postes` | inline `excel_generator` → appeler methode existante |

### Strategie : Strangler Fig (coherent avec l'archi existante)

1. Creer package avec `__init__.py` qui re-exporte tout (zero breaking change)
2. Extraire un module a la fois, feuilles d'abord (config, prompts, filters)
3. Un commit par extraction, testable independamment
4. Decomposer les methodes geantes en steps nommes

---

## 6. Ordre recommande

1. **Fixes critiques** (CRIT-1 a CRIT-7) — rapide, impact securite/bugs
2. **Split `devis_service.py`** — plus gros gain maintenabilite
3. **Split `search.py`** — meme pattern
4. **Eliminer duplications** (BM25, _norm, _synthesize, Ollama)
5. **Refacto frontend** — hooks + memo + cleanup
6. **Fixes medium restants** — perf, qualite
