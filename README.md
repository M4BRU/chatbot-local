# Assistant RAG Multi-Collections

Chatbot RAG 100% local avec support multi-collections et multi-formats (PDF, DOCX, TXT, MD, CSV).

Utilise **Ollama** (llama3.2:3b), **ChromaDB** et **Streamlit** pour fournir des réponses basées sur vos documents, avec indexation incrémentale et isolation par collection.

## Prérequis

- **Python 3.12+**
- **Ollama** installé avec le modèle `llama3.2:3b`

## Installation

```bash
cd ~/chatbot-local
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Télécharger le modèle Ollama (si pas déjà fait)
ollama pull llama3.2:3b
```

### OCR (optionnel)

Pour les PDFs scannés (images), installer Poppler et Tesseract :

```bash
sudo apt install poppler-utils tesseract-ocr tesseract-ocr-fra
pip install pdf2image pytesseract Pillow
```

## Utilisation

### 1. Lancer Ollama

```bash
ollama serve
```

### 2. Indexer des documents (CLI)

```bash
# Indexer un dossier entier dans une collection
python ingest.py vlm_robotics ./documents/

# Indexer un fichier unique
python ingest.py ma_collection ./mon_fichier.pdf

# Forcer la ré-indexation
python ingest.py vlm_robotics ./documents/ --force
```

L'indexation est **incrémentale** : les fichiers déjà indexés (même hash SHA256) sont ignorés automatiquement.

### 3. Lancer l'application

```bash
streamlit run streamlit_app/app.py
```

L'interface s'ouvre dans le navigateur (par défaut http://localhost:8501).

Depuis l'UI vous pouvez :
- Sélectionner une collection
- Créer de nouvelles collections
- Uploader des fichiers directement
- Poser des questions avec réponses en streaming + sources

## Formats supportés

| Format | Extensions | Méthode |
|--------|-----------|---------|
| PDF | `.pdf` | PyPDF2 + OCR fallback |
| Word | `.docx` | python-docx |
| Texte | `.txt`, `.md` | Lecture directe |
| CSV | `.csv` | pandas |

## Architecture

```
chatbot-local/
├── core/                       # Backend modules
│   ├── __init__.py             # Exports
│   ├── embeddings.py           # Config Ollama + OllamaEmbeddings
│   ├── parsers.py              # Parsers multi-format
│   ├── collection_manager.py   # CRUD collections ChromaDB
│   ├── document_manager.py     # Indexation incrémentale (SHA256)
│   └── search.py               # RAGEngine (recherche + génération)
├── streamlit_app/
│   └── app.py                  # Interface Streamlit multi-collections
├── documents/                  # Fichiers sources
├── chroma_db/                  # Base vectorielle (1 sous-dossier par collection)
│   ├── vlm_robotics/
│   │   ├── chroma.sqlite3
│   │   └── metadata.json
│   └── autre_collection/
├── ingest.py                   # CLI d'indexation
├── requirements.txt
└── README.md
```

## Multi-collections

Chaque collection est isolée dans `./chroma_db/{nom}/` avec :
- Sa propre base ChromaDB (`chroma.sqlite3`)
- Son propre fichier de tracking (`metadata.json`) contenant les hash SHA256, dates et chunk_ids
- Son propre historique de conversation dans l'UI

## Prompts personnalisés

Créez un fichier `prompts.json` à la racine pour ajouter des prompts personnalisés :

```json
{
    "mon_prompt": "Tu es un assistant... Contexte : {context}\nQuestion : {question}\n..."
}
```

La collection `vlm_robotics` utilise automatiquement un prompt spécialisé VLM Robotics.

## Troubleshooting

### Ollama ne répond pas

```bash
curl http://localhost:11434
ollama serve
```

### Ré-indexer une collection

```bash
python ingest.py vlm_robotics ./documents/ --force
```

### Supprimer une collection et repartir de zéro

```bash
rm -rf chroma_db/vlm_robotics/
python ingest.py vlm_robotics ./documents/
```

### Réponses lentes

Le modèle tourne sur CPU. Pour accélérer, envisagez un modèle plus léger ou un GPU.
