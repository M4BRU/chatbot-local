# Quickstart - Lancer le Chatbot

Guide rapide pour démarrer le chatbot RAG avec Docker.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│  Frontend   │────▶│   Backend   │────▶│   ChromaDB    │     │    Ollama    │
│  (Next.js)  │     │  (FastAPI)  │────▶│  (HTTP svc)   │     │  (LLM+emb)  │
│ :3000       │     │  :8000      │     │  :8100        │     │  :11434      │
└─────────────┘     └─────────────┘     └───────────────┘     └──────────────┘
                          │                     ▲
                          └─────────────────────┘
                    ingest.py (CLI local) → localhost:8100
```

- **ChromaDB** tourne comme un service Docker indépendant — les données persistent dans le volume `chroma_data` même si le backend redémarre
- **ingest.py** (CLI local) se connecte à ChromaDB via `localhost:8100`
- **Backend** (dans Docker) se connecte à ChromaDB via `chromadb:8000` (réseau Docker interne)
- **Métadonnées d'indexation** (metadata.json) stockées dans le volume `documents_store` (`/app/documents/metadata/`)

## Prérequis

- Docker et Docker Compose installés
- GPU NVIDIA (optionnel, pour de meilleures performances)

## Démarrage rapide

### 1. Lancer l'application

```bash
docker-compose up -d
```

Cette commande lance tous les services:
- **Frontend** (Next.js) sur http://localhost:3000
- **Backend** (FastAPI) sur http://localhost:8000
- **Ollama** (LLM) sur http://localhost:11434
- **ChromaDB** (vector store) sur http://localhost:8100

### 2. Télécharger le modèle LLM

Première utilisation uniquement:

```bash
docker-compose exec ollama ollama pull llama3.2:3b
```

### 3. Accéder à l'interface

Ouvrez votre navigateur sur http://localhost:3000

## Utilisation

1. **Créer une collection** - Sélectionnez ou créez une collection pour vos documents
2. **Uploader des documents** - Formats supportés: PDF, DOCX, TXT, MD, CSV, XLSX, XLS (sélection multiple possible)
3. **Poser des questions** - Le chatbot répond en se basant sur vos documents

## Changer de modèle LLM

Par défaut, le chatbot utilise `llama3.1:8b`. Voici comment passer à un autre modèle (par exemple `llama3.2:3b` ou `llama3.2:8b`).

### Étape 1: Lister les modèles disponibles

```bash
# Voir les modèles déjà téléchargés
docker-compose exec ollama ollama list

# Résultat exemple:
# NAME              ID            SIZE      MODIFIED
# llama3.1:8b       abc123...     4.7 GB    2 days ago
```

### Étape 2: Télécharger le nouveau modèle

```bash
# Télécharger llama3.2:3b (plus léger et rapide)
docker-compose exec ollama ollama pull llama3.2:3b

# OU télécharger llama3.2:8b (plus performant mais plus lourd)
docker-compose exec ollama ollama pull llama3.2:8b

# OU d'autres modèles disponibles
docker-compose exec ollama ollama pull mistral:7b
docker-compose exec ollama ollama pull phi3:medium
```

### Étape 3: Configurer le backend

**Option A: Via variables d'environnement (recommandé)**

Créez un fichier `.env` à la racine du projet:

```bash
# .env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

**Option B: Modifier directement les fichiers**

Modifiez `backend/config/settings.py` ligne 22:

```python
ollama_model: str = "llama3.2:3b"  # Changez ici
```

ET modifiez `backend/core/embeddings.py` ligne 12:

```python
OLLAMA_MODEL = "llama3.2:3b"  # Changez ici aussi
```

### Étape 4: Redémarrer le backend

```bash
# Si vous utilisez .env (option A)
docker-compose down
docker-compose up -d

# Si vous avez modifié les fichiers (option B)
docker-compose down
docker-compose up -d --build backend
```

### Étape 5: Vérifier le changement

```bash
# Vérifier les logs pour confirmer le modèle utilisé
docker-compose logs backend | grep -i model

# Tester via l'interface web
# Ouvrez http://localhost:3000 et posez une question
```

### Supprimer un ancien modèle (libérer de l'espace)

```bash
# Lister les modèles
docker-compose exec ollama ollama list

# Supprimer un modèle spécifique
docker-compose exec ollama ollama rm llama3.1:8b

# Vérifier l'espace libéré
docker-compose exec ollama ollama list
```

### Comparaison des modèles populaires

| Modèle | Taille | Vitesse | Performance | Usage recommandé |
|--------|--------|---------|-------------|------------------|
| `llama3.2:3b` | ~2 GB | Très rapide | Bonne | Tests, prototypage rapide |
| `llama3.2:8b` | ~4.7 GB | Rapide | Très bonne | Production, balance vitesse/qualité |
| `llama3.1:8b` | ~4.7 GB | Rapide | Très bonne | Production standard |
| `mistral:7b` | ~4.1 GB | Rapide | Excellente | Production, excellente qualité |
| `phi3:medium` | ~7.9 GB | Moyenne | Excellente | Qualité maximale |

### Modèle d'embeddings

Le modèle d'embeddings (`nomic-embed-text`) est utilisé pour vectoriser les documents. Il est recommandé de **ne pas le changer** sauf si vous savez ce que vous faites, car cela nécessiterait de réindexer tous vos documents.

Pour changer quand même:

```bash
# Télécharger un autre modèle d'embeddings
docker-compose exec ollama ollama pull mxbai-embed-large

# Modifier .env
OLLAMA_EMBED_MODEL=mxbai-embed-large

# Supprimer ChromaDB et réindexer TOUS vos documents
docker-compose down
docker volume rm chatbot-local_chroma_data
docker-compose up -d
```

## Commandes utiles

### Voir les logs
```bash
docker-compose logs -f
docker-compose logs -f backend    # Backend seulement
docker-compose logs -f frontend   # Frontend seulement
```

### Arrêter l'application
```bash
docker-compose down
```

### Redémarrer l'application
```bash
docker-compose restart
```

### Supprimer les données et repartir de zéro
```bash
docker-compose down -v  # Supprime aussi les volumes (ChromaDB, Ollama)
```

## API Backend

Documentation interactive disponible sur http://localhost:8000/docs

Endpoints principaux:
- `GET /api/v1/health` - Status des services (ollama, chromadb, gpu)
- `GET /api/collections` - Liste des collections
- `POST /api/collections` - Créer une collection
- `GET /api/collections/{name}/documents` - Lister les documents d'une collection
- `POST /api/collections/{name}/documents` - Uploader un document
- `POST /api/chat` - Envoyer un message (streaming SSE)
- `POST /api/chat/sync` - Envoyer un message (réponse JSON complète)

## Debugging — Comprendre ce qu'il se passe

### Logs en temps réel

```bash
# Tous les services à la fois
docker-compose logs -f

# Un seul service (plus lisible)
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f ollama

# Les 50 dernières lignes sans suivre en live
docker-compose logs --tail=50 backend
```

### Tester le backend directement (sans UI)

```bash
# Health check — vérifie ollama, chromadb, gpu
curl http://localhost:8000/api/v1/health

# Lister les collections
curl http://localhost:8000/api/collections

# Lister les documents d'une collection
curl http://localhost:8000/api/collections/vlm/documents

# Poser une question sans streaming (réponse JSON complète)
curl -s -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "Qu est-ce que la machine SOLO ?", "collection_name": "vlm", "prompt_name": "vlm"}' \
  | python3 -m json.tool
```

> La documentation interactive complète est disponible sur **http://localhost:8000/docs** (Swagger UI).

### Vérifier ce qui est dans ChromaDB

ChromaDB tourne comme un service HTTP — on s'y connecte via `HttpClient` :

```bash
# Lancer depuis la racine du projet
venv/bin/python -c "
import chromadb

client = chromadb.HttpClient(host='localhost', port=8100)
print('Collections :', [c.name for c in client.list_collections()])

col = client.get_collection('vlm')
print('Nombre de chunks :', col.count())

# Voir les 3 premiers chunks avec leurs métadonnées
results = col.get(limit=3, include=['metadatas', 'documents'])
for meta, doc in zip(results['metadatas'], results['documents']):
    print('---')
    print('Metadata :', meta)
    print('Extrait  :', doc[:100])
"
```

### Tester Ollama directement

```bash
# Vérifier les modèles disponibles
curl http://localhost:11434/api/tags | python3 -m json.tool

# Tester une génération (hors RAG)
curl -s http://localhost:11434/api/generate \
  -d '{"model": "llama3.1:8b", "prompt": "Dis bonjour en une phrase.", "stream": false}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['response'])"

# Tester le modèle d'embedding
curl -s http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "search_query: test"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('Dimensions:', len(d['embedding']))"
```

### Vérifier que les services se voient entre eux (inside Docker)

```bash
# Le backend peut-il joindre Ollama ?
docker-compose exec backend curl -s http://ollama:11434/api/tags | python3 -m json.tool

# Le backend peut-il joindre ChromaDB ?
docker-compose exec backend python3 -c "import chromadb; print(chromadb.HttpClient(host='chromadb', port=8000).heartbeat())"
```

### Inspecter ce que fait le backend au démarrage

```bash
# Voir les imports et erreurs au lancement
docker-compose logs backend | head -30

# Vérifier que PYTHONPATH est correct dans le conteneur
docker-compose exec backend python3 -c "import sys; print('\n'.join(sys.path))"

# Vérifier quelle version de core/ est utilisée
docker-compose exec backend python3 -c "import core.embeddings; print(core.embeddings.__file__)"
```

---

## Troubleshooting

### Les services ne démarrent pas
```bash
docker-compose down
docker-compose up -d --build
```

### Ollama ne répond pas
```bash
docker-compose restart ollama
docker-compose exec ollama ollama list  # Vérifier les modèles
```

### Réinitialiser ChromaDB
```bash
docker-compose down
docker volume rm chatbot-local_chroma_data
docker-compose up -d
```

## Indexation locale (CLI)

Pour indexer des documents depuis la ligne de commande (sans passer par l'interface web) :

```bash
# Prérequis : Ollama doit tourner (via Docker ou localement)
# Se placer à la racine du projet

# Indexer tous les documents d'un dossier
venv/bin/python ingest.py vlm ./documents/

# Forcer la réindexation (même si déjà présent)
venv/bin/python ingest.py vlm ./documents/ --force

# Indexer un seul fichier
venv/bin/python ingest.py vlm ./documents/SOLO.pdf --force
```

> **Note :** `ingest.py` se connecte à ChromaDB via HTTP (`localhost:8100` par défaut). Pour pointer vers une autre instance, définir les variables d'env `CHROMA_HOST` et `CHROMA_PORT` avant de lancer la commande.
> Les paramètres de chunking (taille, overlap) sont dans `backend/core/document_manager.py`.

---

## Diagnostic du retrieval RAG

Si le chatbot ne trouve pas les bons documents, utilisez le script de diagnostic :

```bash
# Lancer depuis la racine du projet (Ollama doit être actif)
venv/bin/python test_retrieval.py
```

Le script teste 3 questions prédéfinies et affiche pour chaque résultat :
- **Score L2** (distance vectorielle — plus proche de 0 = meilleur)
- **Source** et **métadonnées** du chunk retourné
- **200 premiers caractères** du contenu
- **Diagnostic automatique** avec conseils

### Interpréter les scores (distance L2)

| Score L2 | Cosine similarité | Diagnostic |
|---|---|---|
| < 0.45 | > 90% | Excellent |
| 0.45 – 0.63 | > 80% | Bon |
| 0.63 – 0.85 | > 64% | Moyen — reformuler ou réduire chunk_size |
| > 0.85 | < 64% | Problème d'embedding — réindexer |

> **Rappel :** ChromaDB retourne de la **distance L2**, pas la similarité cosine.
> Formule de conversion : `cos_sim = 1 - L2² / 2`

### Utiliser `debug_retrieval()` dans le code

```python
# Depuis la racine du projet
import sys; sys.path.insert(0, 'backend')
from core.search import RAGEngine

engine = RAGEngine("vlm", prompt_name="vlm")
summary = engine.debug_retrieval("votre question ici", n_results=10)
# Affiche tous les résultats avec scores + diagnostic
```

### Paramètres adaptatifs (K et num_ctx)

K (nombre de chunks envoyés au LLM) et `num_ctx` (taille du contexte Ollama) sont calculés **automatiquement** selon la taille de la collection :

| Nb chunks | K | num_ctx |
|---|---|---|
| < 48 | 6 | 4096 |
| ~200 | 12 | 4096 |
| ~400 | 20 | 6200 |

Ces valeurs sont calculées dans `RAGEngine._adapter_parametres()` (`backend/core/search.py`).

---

## Rebuilder après modification du code

Quand tu modifies un fichier Python ou TypeScript, Docker ne le prend pas en compte automatiquement (les fichiers sont copiés à l'image au moment du build).

```bash
# Modifier du code backend (Python)
docker-compose build backend
docker-compose up -d backend

# Modifier du code frontend (TypeScript/TSX)
docker-compose build frontend
docker-compose up -d frontend

# Modifier les deux
docker-compose build backend frontend
docker-compose up -d backend frontend
```

> **Exception :** Les variables d'environnement dans `.env` sont lues au démarrage — un simple `docker-compose up -d` suffit sans rebuild.

---

## Mode développement

### Frontend
```bash
cd frontend
npm install
npm run dev  # http://localhost:3000
```

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py  # http://localhost:8000
```

N'oubliez pas de lancer Ollama et ChromaDB séparément en mode dev.
