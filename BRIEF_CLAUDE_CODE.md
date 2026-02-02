# 🤖 Brief pour Claude Code - Chatbot RAG Local VLM Robotics

## 📋 Contexte

Je veux créer un chatbot RAG 100% local pour assister sur des tâches commerciales liées à des machines robotiques (bras robotiques, fabrication additive).

### Environnement actuel

**Système** : WSL2 Ubuntu sur Windows  
**Dossier projet** : `~/chatbot-local`  
**Python** : 3.12 (venv activé dans `./venv`)  
**Modèle LLM** : Ollama avec `llama3.2:3b` (déjà installé et fonctionnel)

### Packages déjà installés
```bash
ollama
chromadb
langchain
langchain-community
streamlit
PyPDF2
python-docx
pdf2image
pytesseract
Pillow
```

### Documents à indexer

**Emplacement** : `~/chatbot-local/documents/`  
**Type** : 4 PDFs de brochures techniques VLM Robotics
- Plaquette-FR-GEMINI.pdf
- Plaquette-SOLO-FR.pdf
- Plaquette-FR-COMPAQT.pdf
- PLAQUETTE-HYMANCO-DEF.pdf

**Contenu** : Specs techniques, dimensions, capacités, applications des machines

---

## 🎯 Objectifs

### 1. Script d'ingestion : `ingest.py`

**Fonctionnalités requises :**
- Lire tous les PDFs du dossier `./documents`
- Extraire le texte (PyPDF2 + OCR si nécessaire avec pytesseract)
- Chunking intelligent avec `RecursiveCharacterTextSplitter`
  - chunk_size : 500-800 caractères
  - chunk_overlap : 50-100 caractères
- Créer des embeddings avec Ollama (`llama3.2:3b`)
- Stocker dans ChromaDB local (`./chroma_db`)
- Afficher une barre de progression
- Gérer les erreurs proprement

**Exemple d'utilisation :**
```bash
python ingest.py
# Output : "✅ 4 documents indexés, 127 chunks créés"
```

---

### 2. Application Streamlit : `app.py`

**Interface requise :**
- Design clean et pro (pas trop de couleurs)
- Titre : "Assistant Commercial VLM Robotics"
- Zone de chat avec historique de conversation
- Input utilisateur en bas

**Fonctionnalités :**
- Charger ChromaDB au démarrage
- Pour chaque question :
  1. Recherche de similarité (top k=3-5 chunks)
  2. Construction du prompt avec contexte
  3. Appel à Ollama `llama3.2:3b` via API
  4. Streaming de la réponse si possible
  5. Affichage des sources utilisées (nom du fichier + page si dispo)
- Bouton "Effacer l'historique"
- Gestion des erreurs (Ollama non démarré, ChromaDB vide, etc.)

**Prompt système suggéré :**
```
Tu es un assistant commercial expert pour VLM Robotics. 
Tu aides à rédiger des offres et renseigner les clients sur nos machines de fabrication additive et robotique.

Contexte disponible :
{context}

Question client : {question}

Réponds de manière professionnelle, précise et concise. 
Cite toujours la source (nom de la machine/brochure).
Si l'info n'est pas dans le contexte, dis-le clairement.
```

---

### 3. README.md

**Contenu :**
- Description du projet
- Prérequis
- Installation
- Utilisation
  - Comment lancer Ollama
  - Comment indexer les documents
  - Comment lancer l'app
- Exemples de questions à poser
- Troubleshooting

---

## 🔐 Contraintes importantes

1. **100% local** - Aucun appel API externe (sauf Ollama local)
2. **Données sécurisées** - Tout reste sur la machine
3. **Performance** - Doit fonctionner sur CPU (pas de GPU requis)
4. **Simplicité** - Code lisible et maintenable
5. **Français** - Interface et prompts en français

---

## 📝 Structure de fichiers attendue
```
~/chatbot-local/
├── venv/                    # Environnement virtuel (existant)
├── documents/               # PDFs sources (existant)
│   ├── Plaquette-FR-GEMINI.pdf
│   ├── Plaquette-SOLO-FR.pdf
│   ├── Plaquette-FR-COMPAQT.pdf
│   └── PLAQUETTE-HYMANCO-DEF.pdf
├── chroma_db/              # Base vectorielle (à créer)
├── ingest.py               # Script d'indexation (à créer)
├── app.py                  # Interface Streamlit (à créer)
├── README.md               # Documentation (à créer)
└── requirements.txt        # Dépendances (optionnel)
```

---

## 🧪 Exemples de questions à supporter

- "Quels sont les différents modèles de machines disponibles ?"
- "Quelle est la capacité maximale du modèle GEMINI ?"
- "Compare SOLO et COMPAQT en termes de dimensions"
- "Quelle machine recommandes-tu pour fabriquer une pièce de 4 mètres ?"
- "Qu'est-ce que la technologie WAAM ?"
- "Aide-moi à rédiger une offre pour un client qui veut faire de la fabrication additive XXL"

---

## ⚡ Points d'attention

- **Ollama doit être lancé** : `ollama serve` en arrière-plan
- **Vérifier la connexion** : http://localhost:11434
- **Embeddings** : Utiliser le même modèle que le LLM (`llama3.2:3b`)
- **Gestion mémoire** : Attention au context window (~8k tokens pour llama3.2)
- **Erreurs communes** :
  - Ollama pas démarré
  - ChromaDB vide (pas d'ingestion faite)
  - PDFs corrompus

---

## 🚀 Commandes pour tester après création
```bash
# 1. Activer l'environnement
source venv/bin/activate

# 2. Vérifier Ollama
ollama list

# 3. Indexer les documents
python ingest.py

# 4. Lancer l'app
streamlit run app.py
```

---

## 📊 Métriques de succès

✅ Ingestion de 4 PDFs en < 2 minutes  
✅ Réponses pertinentes avec sources citées  
✅ Interface fluide et réactive  
✅ Pas de crash même si Ollama redémarre  
✅ Code propre et commenté

---

## 💡 Bonus (si le temps le permet)

- Possibilité d'ajouter des PDFs sans tout réindexer
- Export des conversations en PDF
- Statistiques d'utilisation
- Mode "comparaison de produits" optimisé
- Templates d'offres commerciales

---

**Crée les fichiers `ingest.py`, `app.py` et `README.md` en suivant ces specs. Merci !**
