# Idées Data Science applicables au projet

_Notes du 2026-03-02_

## Résumé priorisé

| Priorité | Principe | Impact | Effort |
|----------|----------|--------|--------|
| 🔴 1 | **RAGAS (évaluation)** | Mesure objective des améliorations | Moyen |
| 🔴 2 | **Semantic cache** | UX + latence | Faible |
| 🟠 3 | **Feedback loop + LTR** | Amélioration continue | Élevé |
| 🟠 4 | **BERTopic** | Qualité collections | Moyen |
| 🟡 5 | **Query analytics** | Vision métier | Faible |
| 🟡 6 | **Score calibration** | Robustesse | Moyen |
| 🟢 7 | **Anomaly detection catalogue** | Fiabilité devis | Faible |

---

## 1. Evaluation framework — RAGAS (priorité critique)

**Problème actuel** : `test_retrieval.py` est qualitatif. Aucune métrique objective pour comparer deux configs de retrieval.

**Solution** : RAGAS — framework d'évaluation RAG sans ground truth.

Métriques clés :
- `faithfulness` → le LLM invente-t-il des faits ?
- `answer_relevancy` → la réponse répond-elle à la question ?
- `context_precision` → les chunks récupérés sont-ils pertinents ?
- `context_recall` → les chunks couvrent-ils toute l'info nécessaire ?

**Pourquoi urgent** : plusieurs knobs configurables (`CHUNK_SIZE_TOKENS`, `USE_RERANKER`, `USE_HYBRID_SEARCH`, seuils BM25...) mais aucun moyen rigoureux de savoir si un changement améliore ou dégrade la qualité.

---

## 2. Semantic cache

**Problème** : chaque requête relance embedding + retrieval + LLM (~5–15s).

**Principe** : si deux requêtes sont sémantiquement proches (cosine > 0.92), retourner la réponse en cache.

```python
# Pattern : mini ChromaDB de cache
# query → embed → chercher dans cache_db → hit ou miss
# Seuil calibré sur distribution réelle des requêtes
```

**Impact** : latence quasi-nulle sur les questions récurrentes (techniciens posent souvent les mêmes questions).

---

## 3. Feedback loop implicite + Learning-to-Rank

**Principe** : transformer les interactions en signal d'apprentissage.

- 👍 / 👎 sur une réponse → log `{question, chunks_récupérés, label}`
- Sources cliquées → signal de pertinence positif

Ces données permettent ensuite :
- **Fine-tuning du cross-encoder** (BGE reranker) sur des paires (query, chunk) métier
- **Learning-to-Rank** : remplacer le RRF fixe (k=60) par un modèle appris sur vos données

Note : le cross-encoder `bge-reranker-v2-m3` est déjà en place — le fine-tuner sur 200–300 exemples labelisés suffirait à des gains significatifs.

---

## 4. BERTopic — organisation automatique des collections

**Problème actuel** : collections créées manuellement. Pas de validation que les documents sont bien regroupés.

**BERTopic** : clustering de documents via embeddings → topics automatiques.

```python
from bertopic import BERTopic
# Entrée : embeddings déjà calculés dans ChromaDB
# Sortie : topics + mots-clés représentatifs
```

**Usage concret** :
- Détecter des documents orphelins mal indexés
- Suggérer des sous-collections si une collection est trop hétérogène
- Analyser la distribution des types de documents (maintenance, installation, sécurité...)

---

## 5. Query analytics

**Principe** : logger et clustériser les questions posées.

```
Question A : "quelle est la pression d'air GEMINI ?"
Question B : "pression pneumatique sur cellule GEMINI ?"
→ Même cluster → FAQ automatique + chunk dédié à créer
```

**Output actionnable** :
- Questions fréquentes sans bonne réponse → lacunes documentaires à combler
- Clusters de questions → identifier les domaines critiques pour les techniciens

---

## 6. Calibration des seuils de score

**Problème actuel** : seuils (`0.01`, `0.45`, `0.63`, `0.85`) sont empiriques et cassent si on change de modèle d'embedding ou de reranker.

**Solution** : Isotonic Regression ou Platt Scaling pour calibrer les scores bruts → probabilités [0,1] interprétables.

```python
from sklearn.calibration import CalibratedClassifierCV
# Apprendre P(pertinent | score) sur des exemples labelisés
```

---

## 7. Anomaly detection sur le catalogue dévis

**Problème** : le catalogue Excel contient des prix. Des erreurs de saisie (0€, 100x le prix normal) sont possibles.

**Solution** : Isolation Forest ou Z-score par catégorie produit.

```python
from sklearn.ensemble import IsolationForest
# Détecte les lignes avec prix aberrant avant import dans SQLite
```
