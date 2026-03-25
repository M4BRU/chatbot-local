# Code Review Complet — 2026-03-23 (v2)

Scope : **~13 500 lignes** sur 22 fichiers backend + frontend.
Reviewer : Claude Opus 4.6 (4 agents paralleles + context7 docs verification)
APIs verifiees via context7 : **FastAPI 0.128**, **Next.js 14/16**, **httpx**, **Qdrant client**, **React 18+**

---

## Methodologie

1. **4 agents Explore** ont analyse les fichiers en parallele (~5 min total)
2. **context7 MCP** a ete utilise pour verifier les API actuelles de FastAPI, httpx, Qdrant, Next.js et React — chaque fix recommande utilise les patterns **non-deprecies** confirmes par la doc officielle
3. Les issues de la v1 ont ete **reverifiees** avec les numeros de ligne exacts

### Patterns confirmes a jour par context7

| Lib | Pattern actuel (non-deprecie) | Source |
|---|---|---|
| FastAPI | `Annotated[User, Depends(get_current_user)]` au lieu de `user=Depends(...)` | fastapi.tiangolo.com/tutorial/security |
| FastAPI | `Security(dep, scopes=[...])` pour auth avec scopes OAuth2 | fastapi.tiangolo.com/reference/dependencies |
| FastAPI | `Depends(dep, scope="request")` pour lifecycle management | fastapi.tiangolo.com/reference/dependencies |
| httpx | **Un seul `AsyncClient` partage** — ne PAS creer un nouveau par appel | encode/httpx docs/async.md |
| httpx | `async with httpx.AsyncClient() as client` OU instance globale avec `await client.aclose()` | encode/httpx docs/api.md |
| Qdrant | `client.query_points()` remplace `client.search()` (deprecie) | qdrant.tech/documentation/concepts |
| Qdrant | Prefetch + multi-vector via `models.Prefetch()` | qdrant.tech/documentation/concepts/hybrid-queries |
| React | `React.memo` avec custom equality pour listes | React 18 docs |
| React | `useRef` pour valeurs stables dans callbacks (eviter stale closures) | React 18 docs |
| Next.js | `'use client'` obligatoire pour useState/useEffect | Next.js App Router docs |

---

## 1. Issues Critiques (CRIT) — A fixer en priorite absolue

### CRIT-1: 5 endpoints sans authentification — `collections.py`

- **Fichier:** `backend/api/routes/collections.py` lignes 45-46, 56-57, 73-74, 113-114, 220-221
- **Endpoints exposes :**
  - `POST /` (create_collection) — ligne 45
  - `GET /{name}` (get_collection) — ligne 56
  - `GET /{name}/sources` (list_sources) — ligne 73
  - `GET /{name}/chunks` (list_chunks) — ligne 113
  - `DELETE /{name}` (delete_collection) — ligne 220
- Seul `list_collections` (ligne 31) a `user=Depends(get_current_user)`.
- **Fix (pattern FastAPI actuel confirme context7) :**
```python
from typing import Annotated
from fastapi import Depends, Security

# Pour les routes admin (create/delete) :
@router.post("", response_model=CollectionInfo, status_code=201)
async def create_collection(
    request: CollectionCreate,
    user: Annotated[UserTable, Depends(get_current_user)]
) -> CollectionInfo:
    # Ajouter check role admin si necessaire

# Pour les routes lecture :
@router.get("/{name}", response_model=CollectionInfo)
async def get_collection(
    name: str,
    user: Annotated[UserTable, Depends(get_current_user)]
) -> CollectionInfo:
```

---

### CRIT-2: SQL Injection via NL2SQL — `excel_collection_adapter.py`

- **Fichier:** `backend/adapters/excel_collection_adapter.py:420-430`
- `_execute_sql` fait seulement `sql.strip().upper().startswith("SELECT")` puis execute le SQL brut.
- **Vecteurs d'attaque :**
  - `SELECT * FROM sqlite_master` — expose le schema
  - `SELECT * FROM xl_data; DROP TABLE xl_data` — multi-statement
  - `SELECT load_extension(...)` — execution de code arbitraire
- **Fix recommande :**
```python
import sqlite3

def _execute_sql(self, sql: str) -> list[dict]:
    """Execute un SELECT read-only avec protections."""
    # 1. Ouvrir en read-only
    conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # 2. Autoriser seulement SELECT sur tables xl_*
    def authorizer(action, arg1, arg2, db_name, trigger):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if arg1 and arg1.startswith("xl_"):
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(authorizer)

    # 3. Forcer LIMIT
    if "LIMIT" not in sql.upper():
        sql = f"{sql.rstrip(';')} LIMIT 200"

    try:
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
```

---

### CRIT-3: CRAG dedup ne fonctionne jamais — `search.py`

- **Fichier:** `backend/core/search.py:1516`
- `r not in fallback_crag` compare par identite objet. `langchain_core.documents.Document` n'a pas de `__eq__` custom, donc la comparaison est toujours `True` (objets differents).
- **Resultat :** les resultats CRAG fallback sont toujours dupliques.
- **Fix :**
```python
def _doc_hash(doc):
    return doc.page_content[:150]

fallback_hashes = {_doc_hash(d) for d, _ in fallback_crag}
resultats = fallback_crag + [r for r in resultats if _doc_hash(r[0]) not in fallback_hashes]
```

---

### CRIT-4: Mauvais CollectionManager dans `chat_sync` — `chat.py`

- **Fichier:** `backend/api/routes/chat.py:253, 258`
- Instancie `CollectionManager()` directement au lieu de `get_collection_manager()`.
- **Consequence :** si `VECTOR_DB=qdrant`, cet endpoint utilise quand meme ChromaDB.
- **Fix :**
```python
# Remplacer ligne 258 :
cm = get_collection_manager()  # Utilise le factory DI (singleton, respecte VECTOR_DB)
```

---

### CRIT-5: Route collision `/{name}` capture `version-status` — `collections.py`

- **Fichier:** `backend/api/routes/collections.py:209 vs 56`
- `GET /version-status` (ligne 209) est defini APRES `GET /{name}` (ligne 56).
- FastAPI matche dans l'ordre : `/{name}` capture "version-status" comme nom de collection.
- **Fix :** Deplacer `/version-status` AVANT toutes les routes `/{name}`.

---

### CRIT-6: `search_docs` parse `chunks` au lieu de `components` — `devis_service.py`

- **Fichier:** `backend/domain/services/devis_service.py:2789-2803`
- `parsed.get("chunks", [])` mais `_execute_tool` retourne `{"components": [...], "sources": [...]}`.
- **Resultat :** la recherche retourne toujours count=0, sources vides. Bug silencieux.
- **Fix :**
```python
components = parsed.get("components", [])
sources = parsed.get("sources", [])
```

---

### CRIT-7: `max(pages)` crash si `None` dans le set — `search.py`

- **Fichier:** `backend/core/search.py:715`
- `pages` contient potentiellement `None` via `metadata.get("page")`.
- `max(pages) - min(pages)` leve `TypeError`.
- **Fix :**
```python
numeric_pages = {p for p in pages if isinstance(p, (int, float))}
if len(sources) > 1 or (len(numeric_pages) > 1 and max(numeric_pages) - min(numeric_pages) > 3):
    return None
```

---

### CRIT-8 (NOUVEAU): `_validate_title_with_ai` fail-open au lieu de fail-safe — `document_manager.py`

- **Fichier:** `backend/core/document_manager.py:443-445`
- Le `except` retourne `True` (= "c'est un titre") quand le LLM echoue.
- Commentaire dit "fail-safe" mais c'est en fait "fail-open" — des non-titres sont marques comme titres.
- **Impact :** Pollution de la hierarchie documentaire, degradation de l'extraction de sections.
- **Fix :**
```python
except Exception as e:
    logger.warning("Title validation AI failed: %s — conservative False", e)
    return False  # Conservateur : en cas de doute, ce n'est PAS un titre
```

---

### CRIT-9 (NOUVEAU): 3x `queue.get()` sans timeout — `orchestrator_service.py`

- **Fichier:** `backend/domain/services/orchestrator_service.py` lignes 258, 333, 391
- `await queue.get()` sans timeout. Si le thread de fond crash, le stream SSE bloque indefiniment.
- **Impact :** Connexions HTTP zombies, fuite de ressources, client bloque sans erreur.
- **Fix (pattern asyncio actuel) :**
```python
try:
    kind, data = await asyncio.wait_for(queue.get(), timeout=120.0)
except asyncio.TimeoutError:
    yield f"data: {json.dumps({'error': 'Timeout — aucune reponse du pipeline'})}\n\n"
    break
```

---

## 2. Issues High — Bugs, securite, performance

| # | Fichier | Ligne(s) | Issue | Fix |
|---|---|---|---|---|
| HIGH-1 | `chat.py` | 68-69 | `except: pass` avale les erreurs de persistence | `logger.error(...)` + ne pas raise |
| HIGH-2 | `devis_service.py` | 2952-2969 | `generate_devis` ne filtre pas `<think>` tags (qwen3) | Ajouter `"think": False` dans le payload JSON + filtre streaming |
| HIGH-3 | `excel_collection_adapter.py` | 409 | Pas de LIMIT sur SQL genere par LLM → OOM | `fetchmany(200)` ou injecter `LIMIT 200` |
| HIGH-4 | `devis_service.py` | 888,954,1027,1169,1495,1578,2862,2952 | **8x** `httpx.AsyncClient()` cree par appel LLM — fuite TCP | **Client partage** (confirme par httpx docs) |
| HIGH-5 | `search.py` | 901-934 | `_keyword_fallback_search` = 18-27 queries sequentielles | Batching ou regex unique |
| HIGH-6 | `search.py` | 540-595 | MMR re-embed des chunks deja embedes (+100-500ms) | Cacher embeddings ou les recuperer du vector DB |
| HIGH-7 | `excel_collection_adapter.py` | 214-228 | Row-by-row INSERT (10 000x plus lent que executemany) | `conn.executemany()` ou `df.to_sql()` |
| HIGH-8 | `chat.py` | 143-144 | `except: pass` sur enqueue_eval — erreurs silencieuses | `logger.warning(...)` |

### Detail HIGH-4 — httpx.AsyncClient partage (confirme context7)

La doc httpx officielle est explicite :
> "For optimal connection pooling, avoid creating multiple client instances within frequently executed loops; instead, use a single scoped client or a global instance."

**Fix recommande :**
```python
# Module-level dans devis_service.py
_HTTP_CLIENT: httpx.AsyncClient | None = None

async def _get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=600.0)
    return _HTTP_CLIENT

# Remplacer partout :
# AVANT: async with httpx.AsyncClient(timeout=600.0) as client:
# APRES: client = await _get_http_client()
```

---

## 3. Issues Medium — Qualite, perf, duplication

| # | Fichier | Ligne(s) | Issue |
|---|---|---|---|
| MED-1 | `devis_service.py` | 2247,2305,2327,2341,2395,2420 | Panier/tasks re-fetch chaque iteration (~24 DB calls inutiles par boucle tool-calling) |
| MED-2 | `devis.py:169` + `documents.py:206` | 169, 206 | `_synthesize()` / `_call_synthesis()` quasi-identiques — extraire service partage |
| MED-3 | `catalog_adapter.py` | 36-102 | BM25/stemmer reimplemente (existe deja dans `bm25_utils.py`) |
| MED-4 | `catalog_adapter.py:61` vs `excel_collection_adapter.py:64` | — | `_norm()` defini differemment dans 2 fichiers |
| MED-5 | Routes variees | — | Import paths `core.*` vs `backend.core.*` inconsistants |
| MED-6 | `dependencies.py:46,62` + `main.py:121-122` | — | Double `init_db()` (idempotent mais redondant) |
| MED-7 | `devis_service.py` | 423, 451 | `import re as _re` dans le body de fonctions (deja importe ligne 27) |
| MED-8 | 3 fichiers | — | Ollama HTTP calls directs au lieu de passer par un port/adapter |
| MED-9 | `collections.py` | 65-68, 109-110 | Exceptions silencieuses sans logging |
| MED-10 | `chat.py` | — | Pas de `import logging` / logger au niveau module |
| MED-11 | `excel_collection_adapter.py` | 475-478 | BM25 charge la table entiere en memoire (pas de LIMIT) |
| MED-12 | `search.py` | 696, 713 | `pages` set melange `None`, `int`, `str` sans validation |

---

## 4. Frontend — Issues React/Next.js

### Critiques Frontend

| # | Ligne(s) | Issue | Fix |
|---|---|---|---|
| **FE-1** | 1323 | God component 2132L, 20+ useState — impossible a tester/maintenir | Extraire hooks `useDevisChat`, `useDevisPanier`, `useAutoScroll`, `useLLMStatus` |
| **FE-2** | 1534-1782 | `handleSend` = 247 lignes, 13 types d'events SSE | Extraire dans `useDevisChat.ts` hook |
| **FE-3** | 96-112 | **Mock data en prod** (VLM-2024-001 hardcode dans SearchWorkspaceModal) | Remplacer par appel API reel ou supprimer |
| **FE-4** | 1781 | `handleSend` recree chaque keystroke (dep `input` dans useCallback) | Utiliser `inputRef.current` au lieu de `input` state |

### High Frontend

| # | Ligne(s) | Issue | Fix |
|---|---|---|---|
| **FE-5** | 586 | `MessageItem` pas memoize — re-render de TOUS les messages a chaque state change | `React.memo` avec comparaison custom `msg.id + msg.content` |
| **FE-6** | 957 | `DevisPanel` pas memoize — filtre/groupe 100+ postes a chaque render | `React.memo` + `useMemo` pour `mainPostes`/`optionPostes` |
| **FE-7** | 742 | `PosteRow` pas memoize — perd le state local (`expanded`, `menuOpen`) | `React.memo` avec comparaison sur item fields |
| **FE-8** | 96-112, 1780-1781 | `eslint-disable-next-line react-hooks/exhaustive-deps` cache des vrais bugs | Fixer les deps au lieu de desactiver le lint |
| **FE-9** (NOUVEAU) | 1471-1483 | Side-effect async dans `setState` updater (settings save) — anti-pattern React Suspense | Deplacer dans `useEffect` avec debounce |

### Medium Frontend

| # | Ligne(s) | Issue | Fix |
|---|---|---|---|
| **FE-10** | 767-774 | Inputs non-controles synces via refs + `document.activeElement` hack | Convertir en inputs controles |
| **FE-11** | 1464-1468 | `handleScroll` defini mais jamais attache au scrollAreaRef | Ajouter `useEffect` avec `addEventListener` + cleanup |
| **FE-12** | 1365, 1367 | Timers refs (`highlightTimerRef`, `settingsSaveTimerRef`) pas nettoyes au unmount | `useEffect` cleanup |
| **FE-13** | 1541-1545 | Recursion guard via ref mutable (`sendDepthRef`) — fragile si erreur | `AbortController` ou state boolean |

---

## 5. Plan de Refactoring

### Phase 0 — Fixes critiques (1-2 jours)

**Ordre :** CRIT-2 (SQL injection) > CRIT-1 (auth) > CRIT-9 (timeout) > CRIT-8 (fail-open) > CRIT-5 (route collision) > CRIT-6 (chunks vs components) > CRIT-7 (max None) > CRIT-3 (CRAG dedup) > CRIT-4 (CM factory)

### Phase 1 — Backend : Split `devis_service.py` (2974 L) → `backend/domain/services/devis/`

| Module | Contenu | ~Lignes |
|---|---|---|
| `prompts.py` | System prompt, tool definitions, JSON schemas | 170 |
| `conflict_detection.py` | Element/column/relevance/affaire conflict | 400 |
| `rfq_planner.py` | Decompose, search, gap detect, synthesize | 700 |
| `guardrails.py` | Multi-action detection, expected tool nudge | 200 |
| `context_compression.py` | Estimate chars, compress tool results | 100 |
| `ollama_client.py` | **httpx.AsyncClient partage** + `_ollama_call()` helper | 80 |
| `devis_service.py` | Orchestrateur (execute_tool, chat_stream, generate) | 800 |

### Phase 2 — Backend : Split `search.py` (2087 L) → `backend/core/search/`

| Module | Contenu | ~Lignes |
|---|---|---|
| `config.py` | Env vars, constants, context brackets | 170 |
| `prompts.py` | PROMPT_DEFAUT, PROMPT_VLM, domain context | 90 |
| `filters.py` | Image filter, source cap, PDF/DOCX dedup | 150 |
| `bm25.py` | BM25 index, cache, stemmer | 100 |
| `fusion.py` | RRF fusion, MMR (+ fix re-embedding) | 70 |
| `reranker.py` | Reranker + ColBERT singletons | 100 |
| `query_transform.py` | Multi-query, HyDE, query rewrite | 180 |
| `retrieval_augment.py` | Tail extension, section, keyword fallback | 300 |
| `engine.py` | RAGEngine (rechercher decompose en steps) | 400 |
| `reasoning.py` | ReasoningEngine | 200 |

### Phase 3 — Eliminer duplications cross-fichiers

| Duplique | Source → Cible |
|---|---|
| BM25/stemmer | `catalog_adapter.py:36-102` → importer `bm25_utils.py` |
| `_norm()` | 2 fichiers → `backend/utils/text.py` |
| `_synthesize()` | `devis.py` + `documents.py` → `backend/domain/services/synthesis_service.py` |
| Ollama HTTP | 3 fichiers → `OllamaPort` (adapter hexagonal) |
| Import paths | Standardiser sur `backend.core.*` partout |

### Phase 4 — Frontend : Refacto `devis/page.tsx` (2132 L)

**Hooks a extraire :**

| Hook | State | Fichier |
|---|---|---|
| `useDevisChat` | messages, isLoading, error, activeToolCalls, rfqPlanning, candidates, handleSend | `hooks/useDevisChat.ts` |
| `useDevisPanier` | postes, devisSettings, highlightedId, isExporting | `hooks/useDevisPanier.ts` |
| `useAutoScroll` | showScrollBtn, scrollAreaRef + event listener | `hooks/useAutoScroll.ts` |
| `useStreamingMessage` | streamingContent, streamingId (evite O(n) map/token) | `hooks/useStreamingMessage.ts` |
| `useLLMStatus` | llmReady, collections, catalogLoaded | `hooks/useLLMStatus.ts` |

**Composants a extraire + memoiser :**

| Composant | Fichier | Memo |
|---|---|---|
| `MessageItem` | `components/MessageItem.tsx` | `React.memo` custom equality |
| `DevisPanel` | `components/DevisPanel.tsx` | `React.memo` + `useMemo` filtres |
| `PosteRow` | `components/PosteRow.tsx` | `React.memo` custom equality |
| `SearchWorkspaceModal` | `components/SearchWorkspaceModal.tsx` | Supprimer mock data |
| `DevisInputBar` | `components/DevisInputBar.tsx` | — |
| `WelcomeScreen` | `components/WelcomeScreen.tsx` | — |

### Strategie : Strangler Fig

1. Creer package avec `__init__.py` qui re-exporte tout (zero breaking change)
2. Extraire un module a la fois, feuilles d'abord (config, prompts, filters)
3. Un commit par extraction, testable independamment
4. Decomposer les methodes geantes en steps nommes

---

## 6. Ordre recommande global

| Priorite | Quoi | Effort | Impact |
|---|---|---|---|
| **P0** | CRIT-2 SQL injection | 1h | Securite critique |
| **P0** | CRIT-1 Auth endpoints | 30min | Securite critique |
| **P0** | CRIT-9 Queue timeout | 30min | Stabilite prod |
| **P0** | CRIT-8 Fail-open title | 5min | Qualite donnees |
| **P1** | CRIT-5,6,7,3,4 (bugs logiques) | 2h | Correctness |
| **P1** | HIGH-4 httpx client partage | 1h | Perf + fuites TCP |
| **P1** | HIGH-2 think tags | 15min | UX |
| **P2** | Phase 1 — Split devis_service.py | 1 jour | Maintenabilite |
| **P2** | Phase 2 — Split search.py | 1 jour | Maintenabilite |
| **P3** | Phase 3 — Eliminer duplications | 0.5 jour | Code quality |
| **P3** | Phase 4 — Refacto frontend | 1-2 jours | Perf + maintenabilite |
| **P4** | MED-1 a MED-12 restants | 1 jour | Perf + qualite |

---

## 7. Resume chiffre

| Severite | Count | Exemples cles |
|---|---|---|
| **CRITIQUE** | 9 | SQL injection, auth manquante, queue timeout, fail-open, route collision, CRAG dedup, max(None), chunks vs components |
| **HIGH** | 8 + 9 FE | httpx leak, think tags, bare except, MMR re-embed, keyword fallback, row-by-row INSERT, React.memo manquants |
| **MEDIUM** | 12 + 4 FE | Duplications, imports, double init_db, panier re-fetch, timer cleanup |
| **Total** | **42 issues** | Backend 29 + Frontend 13 |
