# Audit de code -- chatbot-local RAG backend

**Date :** 2026-02-19
**Auditeur :** Claude Opus 4.6
**Scope :** 18 fichiers (backend Python + frontend TypeScript + infra Docker)

---

## AXE 1 -- Architecture & maintenabilite

### Q1.1 -- Robustesse des imports PYTHONPATH

**[SEVERITE : MAJEUR]** Double resolution PYTHONPATH fragile et source de confusion

- **Localisation :** `backend/Dockerfile:127`, tous les fichiers `core/*.py`
- **Probleme :** L'ordre `PYTHONPATH=/app/backend:/app` signifie que `from core.xxx` resout d'abord dans `/app/backend/core/`. Si un dossier `core/` existe a la racine du projet (ce qui est le cas), l'import fonctionnerait differemment hors Docker. En CI sans Docker, les imports `from core.xxx` echouent sauf si `PYTHONPATH` est configure manuellement. Les tests locaux et le linting (`mypy`, `ruff`) sont non fonctionnels sans configuration supplementaire.
- **Risque concret :** Un developpeur qui clone le projet et lance `pytest` sans Docker obtient `ModuleNotFoundError: No module named 'core'`.
- **Solution :**
  1. **Package installable** (recommande) : `pyproject.toml` avec `pip install -e .`
  2. **Quick fix** : remplacer partout `from core.xxx` par `from backend.core.xxx`
  3. **Minimum pour dev local** : `conftest.py` qui injecte `PYTHONPATH=./backend:.`

---

### Q1.2 -- Respect reel de l'architecture hexagonale

**[SEVERITE : MINEUR]** Ports/adapters declares mais non implementes -- architecture hexagonale fantome

- **Localisation :** `backend/domain/ports/`, `backend/api/dependencies.py:43-47`
- **Probleme :** Les routes importent directement `CollectionManager`, `RAGEngine`, `DocumentManager`. Les ports `LLMPort`, `EmbeddingPort`, `VectorStorePort` ne sont implementes par aucun adapter.
- **Solution :** Pour un POC, acceptable. Ordre d'implementation recommande : `VectorStorePort` → `EmbeddingPort` → `LLMPort`

---

### Q1.3 -- Imports dynamiques dans les handlers

**[SEVERITE : MINEUR]** Lazy imports dans chaque handler -- lisibilite reduite sans gain reel

- **Localisation :** `backend/api/routes/chat.py:37-38`, `collections.py:23`, `documents.py:40-41`
- **Probleme :** Apres le premier import, `sys.modules` met en cache : cout negligeable (~50ns). Intention : eviter circular imports au demarrage. Impact lisibilite : dependances masquees.
- **Solution :** Utiliser `FastAPI Depends()` pour centraliser la creation des objets.

---

### Q1.4 -- Code mort : `_booster_sources_session()` jamais appelee

**[SEVERITE : MINEUR]** Fonction orpheline -- code mort dangereux si branche naivement

- **Localisation :** `backend/core/search.py:331-355`
- **Probleme :** `_booster_sources_session()` est definie mais **jamais appelee** dans le pipeline. C'est du code mort.
- **Solution :** Supprimer OU brancher apres le filtrage par seuil (pas avant, sinon interaction dangereuse avec le seuil relatif).

---

### Q1.5 -- Blocking I/O dans handlers async

**[SEVERITE : MAJEUR]** Appels bloquants dans des coroutines async -- bloque la boucle d'evenements uvicorn

- **Localisation :** `backend/api/routes/chat.py:33-52`, `documents.py:62-75`
- **Probleme :** `_stream_rag_response` itere sur un generateur synchrone `requests.iter_lines()`. Pendant le streaming (5-60s), la boucle d'evenements asyncio est bloquee. Toute autre requete HTTP est mise en attente.
- **Solution :**
  ```python
  # Option 1 : asyncio.to_thread
  result = await asyncio.to_thread(rag.generer_avec_sources, message, stream=False, history=history_text)

  # Option 2 (meilleure) : httpx async pour Ollama
  async with httpx.AsyncClient(timeout=300) as client:
      async with client.stream("POST", OLLAMA_API_GENERATE, json=payload) as resp:
          async for line in resp.aiter_lines():
              data = json.loads(line)
              if data.get("response"):
                  yield data["response"]
  ```

---

## AXE 2 -- Performance & retrieval

### Q2.1 -- `get_embeddings()` appele a chaque requete

**[SEVERITE : MAJEUR]** Instanciation repetee du modele HuggingFace -- rechargement memoire a chaque requete

- **Localisation :** `backend/core/embeddings.py:154-165`, `backend/core/collection_manager.py:209-225`
- **Probleme :** `HuggingFaceEmbeddings` ne fait pas de cache interne global. Chaque instanciation recharge `mxbai-embed-large-v1` (~700 Mo) depuis le disque. ~1-3 secondes de chargement a chaque requete chat.
- **Solution :**
  ```python
  from functools import lru_cache

  @lru_cache(maxsize=1)
  def get_embeddings() -> Embeddings:
      if USE_HF_EMBEDDINGS:
          return HFEmbeddings()
      return NomicEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
  ```

---

### Q2.2 -- Thread-safety des singletons globaux

**[SEVERITE : MINEUR]** Pattern check-then-set sur les singletons -- race condition theorique

- **Localisation :** `backend/core/search.py:548-563`, `search.py:566-579`, `parsers.py:1236-1273`
- **Probleme :** Deux threads pourraient passer le check simultanement et charger le modele deux fois. Impact reel : pic memoire temporaire. Pas de corruption.
- **Solution :** Double-checked locking avec `threading.Lock`.

---

### Q2.3 -- BM25 et race condition pendant ingest

**[SEVERITE : MINEUR]** Index BM25 potentiellement stale pendant l'ingest

- **Localisation :** `backend/core/search.py:433-455`
- **Solution :** Invalider le cache apres ingest : `_bm25_cache.pop(nom_collection, None)`

---

### Q2.4 -- `_adapter_parametres()` : round-trip HTTP ChromaDB a chaque requete

**[SEVERITE : MINEUR]** ~1-5ms par requete pour un count rarement change

- **Localisation :** `backend/core/search.py:678-692`
- **Solution :** Cache TTL de 60 secondes si optimisation necessaire.

---

### Q2.5 -- SemanticChunker : appels embeddings bloquants dans handler async

**[SEVERITE : MAJEUR]** Ingest document bloque la boucle d'evenements pendant plusieurs minutes

- **Localisation :** `backend/api/routes/documents.py:62-75`, `backend/core/document_manager.py:1074-1082`
- **Solution :**
  ```python
  result = await asyncio.to_thread(dm.ajouter_document, collection_name, final_path, force=force)
  ```

---

### Q2.6 -- Query rewriting sans condition de court-circuit

**[SEVERITE : MINEUR]** Appel Ollama systematique meme pour un historique minimal

- **Localisation :** `backend/core/search.py:612-644`
- **Solution :**
  ```python
  nb_echanges = history.count("User:")
  if nb_echanges <= 1:
      return question  # pas assez de contexte pour rewriter
  ```

---

## AXE 3 -- Robustesse & fiabilite

### Q3.1 -- Propagation des exceptions dans `_stream_tokens()`

**[SEVERITE : MAJEUR]** `json.JSONDecodeError` et `ChunkedEncodingError` non catchees, `Response` non fermee

- **Localisation :** `backend/core/search.py:822-832`
- **Solution :**
  ```python
  def _stream_tokens():
      try:
          for ligne in reponse.iter_lines():
              if ligne:
                  try:
                      donnees = json.loads(ligne)
                  except json.JSONDecodeError:
                      continue
                  token = donnees.get("response", "")
                  if token:
                      yield token
                  if donnees.get("done", False):
                      break
      except requests.exceptions.ChunkedEncodingError:
          yield "\n\n[Connexion a Ollama interrompue]"
      finally:
          reponse.close()
  ```

---

### Q3.2 -- Vulnerabilite path traversal dans `upload_document`

**[SEVERITE : CRITIQUE]** `file.filename` non sanitize -- path traversal possible

- **Localisation :** `backend/api/routes/documents.py:62-83`
- **Probleme :** Si `file.filename = "../../etc/cron.d/malicious"`, `final_path` pointe hors du repertoire temporaire.
- **Solution :**
  ```python
  from pathlib import PurePosixPath
  safe_name = PurePosixPath(file.filename or "document").name
  if not safe_name or safe_name == ".":
      safe_name = "document"
  final_path = tmp_path.parent / f"{tmp_path.stem}_{safe_name}"
  ```

---

### Q3.3 -- Fingerprint de deduplication `doc.page_content[:150]`

**[SEVERITE : MINEUR]** Collision possible sur des chunks avec en-tete commun

- **Localisation :** `backend/core/search.py:460`, `search.py:522`
- **Solution :** `hashlib.md5(doc.page_content.encode()).hexdigest()` comme fingerprint.

---

### Q3.4 -- Session scoping + seuil relatif : interaction problematique

**[SEVERITE : MINEUR]** `_booster_sources_session()` non branchee -- risque nul en l'etat

- **Localisation :** `backend/core/search.py:338-355`, `search.py:733-736`
- **Solution :** Appliquer le boost APRES le filtrage par seuil si branche.

---

### Q3.5 -- Race condition dans `_charger_metadata` lors d'ingests paralleles

**[SEVERITE : MINEUR]** Read-modify-write sans verrou sur le fichier JSON metadata

- **Localisation :** `backend/core/document_manager.py:1011-1020`
- **Solution :** `threading.Lock` par collection ou migration vers SQLite.

---

### Q3.6 -- `lru_cache(maxsize=1)` sur `_get_tokenizer()` : thread-safety

**[SEVERITE : MINEUR]** Pas de probleme reel -- `lru_cache` utilise un `threading.RLock` interne.

---

### Q3.7 -- Frontend bloque si warmup echoue

**[SEVERITE : MAJEUR]** Polling infini sans timeout -- UX bloquee indefiniment si Ollama crash

- **Localisation :** `frontend/app/components/Chat.tsx` (useEffect polling)
- **Solution :**
  ```typescript
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    let attempts = 0;
    const MAX_ATTEMPTS = 100; // 5 minutes

    const check = async () => {
      attempts++;
      const ready = await fetchLLMStatus();
      if (ready) {
        setLlmReady(true);
        clearInterval(interval);
      } else if (attempts >= MAX_ATTEMPTS) {
        setError("Le modele IA n'a pas pu demarrer apres 5 minutes.");
        clearInterval(interval);
      }
    };
    check();
    interval = setInterval(check, 3000);
    return () => clearInterval(interval);
  }, []);
  ```

---

## AXE 4 -- Qualite du code

### Q4.1 -- Variables globales lues a l'import-time

**[SEVERITE : MAJEUR]** Double source de verite entre `embeddings.py` et `config/settings.py`

- **Localisation :** `backend/core/embeddings.py:81-84`, `backend/config/settings.py`
- **Probleme :**
  1. Defaults differents entre les deux fichiers (`localhost` vs `ollama`)
  2. Variables non patchables en test apres import
  3. `CORS_ORIGINS` dans docker-compose est une string JSON brute -- probablement mal parsee par pydantic-settings
- **Solution :** Unifier sur `Settings` comme source unique. Ajouter un `@field_validator("cors_origins", mode="before")` qui parse le JSON.

---

### Q4.2 -- `_appeler_ollama` comme staticmethod

**[SEVERITE : MINEUR]** Acceptable -- mock possible via `patch("core.search.RAGEngine._appeler_ollama")`

---

### Q4.3 -- Creation de CollectionManager et RAGEngine a chaque requete

**[SEVERITE : MAJEUR]** Nouvelle connexion TCP ChromaDB a chaque requete

- **Localisation :** `backend/api/routes/chat.py:40-45`
- **Solution :** Singleton `CollectionManager` via `@lru_cache(maxsize=1)` dans `dependencies.py`.

---

### Q4.4 -- Warmup LLM avec `timeout=None`

**[SEVERITE : MAJEUR]** Thread zombie si Ollama ne repond jamais + modele hardcode

- **Localisation :** `backend/main.py`
- **Probleme :** `timeout=None` + `"llama3.1:8b"` hardcode au lieu de `OLLAMA_MODEL`.
- **Solution :** `timeout=600` + utiliser `OLLAMA_MODEL` depuis les settings.

---

### Q4.5 -- Memory leaks potentiels

**[SEVERITE : MINEUR]** `requests.Response` non fermee dans `_stream_tokens()`

- **Solution :** `reponse.close()` dans un `finally`. Singleton `CollectionManager` (cf Q4.3).

---

### Q4.6 -- Domain ports non implementes

**[SEVERITE : MINEUR]** Interfaces vides -- dette documentaire acceptable pour un POC.

---

### Q4.7 -- Couverture de tests quasi-nulle

**[SEVERITE : MAJEUR]** 5 tests critiques a implementer en priorite

1. `parser_document` pour chaque format (2h)
2. Cycle de vie document avec ChromaDB en memoire (2h)
3. `RAGEngine.rechercher` avec ChromaDB pre-rempli (3h)
4. `_rrf_fusion` -- test unitaire pur (30min)
5. `upload_document` endpoint avec TestClient (2h)

---

### Q4.8 -- Bug potentiel : promptName = collectionName dans Chat.tsx

**[SEVERITE : MINEUR]** Convention deliberee -- fallback silencieux vers le prompt par defaut.

---

## TABLEAU DE PRIORISATION

| Priorite | Amelioration | Severite | Effort | Impact |
|----------|-------------|---------|--------|--------|
| 1 | **Singleton `get_embeddings()` avec `@lru_cache`** (Q2.1) | MAJEUR | 15 min | Elimine le rechargement du modele HF (~700 Mo) a chaque requete. Gain de 1-3s de latence par requete chat. |
| 2 | **Sanitize `file.filename` dans `upload_document`** (Q3.2) | CRITIQUE | 30 min | Elimine la vulnerabilite path traversal. Securite de base indispensable. |
| 3 | **`asyncio.to_thread` pour les handlers bloquants** (Q1.5, Q2.5) | MAJEUR | 1h | Debloque la boucle d'evenements pendant le chat et l'ingest. |
| 4 | **Timeout warmup LLM (600s + OLLAMA_MODEL)** (Q4.4) | MAJEUR | 15 min | Evite un thread zombie. Shutdown Docker propre. |
| 5 | **Timeout frontend polling LLM status** (Q3.7) | MAJEUR | 30 min | Evite le blocage infini de l'UI si Ollama ne demarre pas. |
| 6 | **Singleton `CollectionManager` via `lru_cache`** (Q4.3) | MAJEUR | 30 min | Elimine la creation d'une connexion TCP ChromaDB a chaque requete. |
| 7 | **Try/except dans `_stream_tokens` + `reponse.close()`** (Q3.1) | MAJEUR | 15 min | Evite le crash du stream SSE sur ligne Ollama malformee. |
| 8 | **5 tests unitaires critiques** (Q4.7) | MAJEUR | 8h | Filet de securite contre les regressions. |

---

*Audit realise le 2026-02-19 par Claude Opus 4.6.*
