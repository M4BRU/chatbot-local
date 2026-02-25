"""
core/search.py — RAGEngine : recherche similarité + génération Ollama streaming.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from core.collection_manager import CollectionManager
from core.embeddings import OLLAMA_API_GENERATE, OLLAMA_MODEL

# Qwen3 : désactive le mode raisonnement (chain-of-thought) pour les réponses RAG.
# Sans ce flag, Qwen3 génère un bloc <think>...</think> avant chaque réponse (+latence).
ENABLE_NO_THINK = os.environ.get("ENABLE_NO_THINK", "false").lower() == "true"

logger = logging.getLogger(__name__)

# Prompt par défaut générique
PROMPT_DEFAUT = """Tu es un assistant intelligent. Utilise le contexte ci-dessous pour répondre à la question.

Contexte :
{context}

Question : {question}

Réponds de manière précise et concise. Si l'information n'est pas dans le contexte, dis-le clairement."""

# Prompt spécialisé VLM Robotics
PROMPT_VLM_ROBOTICS = """Tu es un assistant commercial expert pour VLM Robotics, constructeur de machines-outils robotisées pour la fabrication hybride XXL (Machine Tool Builder for XXL Hybrid Manufacturing).

Ton expertise couvre :
- La gamme complète : COMPAQT (XL, entrée de gamme), SOLO (XXL mono-robot), GEMINI (XXL bi-robot, la plus avancée), HYMANCO (unité mobile containerisée)
- Les multiples technologies intégrées : WAAM (arc électrique CMT Fronius), fabrication additive hybride laser (poudre et fil), Cold Spray, FSW, usinage, CND, scan/métrologie, collage, polymère FDM
- L'expertise VLM : Direct Control, commande numérique CNC Siemens, logiciel NX (CAO, CAM, jumeau numérique), continuité numérique, Industry 4.0
- Le positionnement hybride : les machines combinent plusieurs procédés sur une même plateforme (ex : fabrication additive + usinage + contrôle)
- Les secteurs : ASD, Ferroviaire, Naval, Énergie, MRO, Fonderie, Outillage, Formation, Recherche, Offshore

Contexte disponible :
{context}

Question client : {question}

Consignes de réponse :
- Réponds en français, de manière professionnelle et structurée.
- Cite toujours la source (nom de la machine, référence brochure, numéro de page).
- Si le contexte permet de recommander une machine spécifique, explique pourquoi elle convient au besoin.
- Si l'information n'est pas dans le contexte fourni, dis-le clairement : « Je n'ai pas trouvé cette information dans la documentation disponible. »
- Ne jamais inventer de spécifications techniques.
- Reproduis les listes du contexte de manière exhaustive : cite TOUS les éléments sans en omettre, abréger ou résumer aucun.
- N'ajoute jamais d'explications ou de phrases qui ne sont pas explicitement dans le contexte fourni."""

# Prompts nommés disponibles
PROMPTS = {
    "defaut": PROMPT_DEFAUT,
    "vlm_robotics": PROMPT_VLM_ROBOTICS,
    "vlm": PROMPT_VLM_ROBOTICS,  # alias : collection "vlm" → prompt VLM Robotics
}

NB_CHUNKS_RECHERCHE = 6   # fallback si collection vide
CHUNK_SIZE_APPROX = 1000  # doit correspondre à document_manager.CHUNK_SIZE
NUM_CTX_MIN = 4096   # 4096 : Qwen3:8b a 37 layers (~4.5 Go VRAM) + KV cache q8_0 306 Mo = ~5.0 Go GPU
                     # 8192 ctx = 612 Mo KV → dépasse les ~5.4 Go dispo sur RTX 3060 Laptop 6 Go
NUM_CTX_MAX = 32768

# ── Metadata filtering ────────────────────────────────────────────────────────
_MACHINES_CONNUES = ["GEMINI", "SOLO", "COMPAQT", "HYMANCO"]

# ── Reranker config ───────────────────────────────────────────────────────────
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_CANDIDATS_MULT = 3   # récupère k×3 candidats avant reranking
RERANKER_CANDIDATS_MAX = 25   # plafond pour éviter un contexte trop large
_reranker_instance = None

# ── ColBERT config (RAGatouille) ──────────────────────────────────────────────
# Late-interaction : scoring token-à-token, meilleur que cross-encoder BGE
# sur les termes techniques et acronymes.
# USE_COLBERT=true remplace BGE. Modèle ~2GB, téléchargé au premier appel.
USE_COLBERT = os.environ.get("USE_COLBERT", "false").lower() == "true"
COLBERT_MODEL = os.environ.get("COLBERT_MODEL", "colbert-ir/colbertv2.0")
_colbert_instance = None

def _deduplicater_pdf_docx(resultats: list) -> list:
    """
    Élimine les doublons entre versions PDF et DOCX du même document.

    Si le même document a été ingéré en deux formats (ex : offre.pdf + offre.docx),
    seul le chunk au score le plus élevé pour chaque (nom_de_base, page) est conservé.

    Exemple :
      "0093 - SAFRAN.pdf" p.3  score 0.91  ←  conservé
      "0093 - SAFRAN.docx" p.3 score 0.87  ←  éliminé (doublon, score inférieur)
      "0093 - SAFRAN.pdf" p.5  score 0.76  ←  conservé (page différente, pas un doublon)
    """
    seen: dict[tuple, tuple[float, int]] = {}  # (base, page) → (score, index)

    for i, (doc, score) in enumerate(resultats):
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "?")
        base = re.sub(r'\.(pdf|docx|doc|txt)$', '', source, flags=re.IGNORECASE)
        key = (base, page)

        if key not in seen or score > seen[key][0]:
            seen[key] = (score, i)

    kept_indices = {idx for _, idx in seen.values()}
    deduped = [r for i, r in enumerate(resultats) if i in kept_indices]

    nb_removed = len(resultats) - len(deduped)
    if nb_removed:
        logger.info(f"Déduplication PDF/DOCX : {nb_removed} doublon(s) supprimé(s)")

    return deduped


# ── Hybrid search (BM25 + vector) ────────────────────────────────────────────
USE_HYBRID_SEARCH = os.environ.get("USE_HYBRID_SEARCH", "true").lower() == "true"
RRF_K = 60  # constante RRF standard (valeur de référence de la littérature)
_bm25_cache: dict[str, tuple[int, object]] = {}  # {collection: (nb_chunks, index)}

# Stemmer français Snowball (NLTK) — normalisé au premier appel
_stemmer = None


def _get_stemmer():
    """Retourne le stemmer Snowball français (lazy init, pas de téléchargement réseau)."""
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem.snowball import FrenchStemmer
            _stemmer = FrenchStemmer()
        except Exception as e:
            logger.warning(f"Stemmer NLTK indisponible ({e}) — stemming désactivé")
    return _stemmer


def _stemmer_tokens(tokens: list[str]) -> list[str]:
    """
    Applique le stemmer Snowball français sur une liste de tokens.
    'directives' → 'direct', 'directive' → 'direct'  ← même racine ✓
    'machines' → 'machin', 'machine' → 'machin'       ← même racine ✓
    Retourne les tokens originaux si le stemmer est indisponible.
    """
    stemmer = _get_stemmer()
    if stemmer is None:
        return tokens
    return [stemmer.stem(t) for t in tokens]


class _BM25CollectionIndex:
    """Index BM25 sur les chunks d'une collection ChromaDB."""

    def __init__(self, docs: list, texts: list):
        from rank_bm25 import BM25Okapi
        self._docs = docs
        # re.findall(r'\w+') split sur apostrophes/tirets/ponctuations
        # → "l'AMDEC" → ["l", "amdec"], "d'Apave" → ["d", "apave"]
        # _stemmer_tokens normalise pluriel/singulier
        # → "directives" et "directive" → même racine "direct"
        tokenized = [_stemmer_tokens(re.findall(r'\w+', t.lower())) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int, filtre: dict | None = None) -> list:
        tokens = _stemmer_tokens(re.findall(r'\w+', query.lower()))
        raw_scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)
        machine_filtre = None
        if filtre:
            machine_filtre = filtre.get("machine", {}).get("$eq")
        results = []
        for i in ranked:
            doc = self._docs[i]
            if machine_filtre and doc.metadata.get("machine") != machine_filtre:
                continue
            results.append((doc, float(raw_scores[i])))
            if len(results) >= k:
                break
        return results


def _get_or_build_bm25(db, collection_name: str) -> "_BM25CollectionIndex | None":
    """
    Retourne l'index BM25 depuis le cache, ou le reconstruit si la collection a changé.
    L'index est invalidé automatiquement quand des documents sont ajoutés/supprimés.
    """
    if not USE_HYBRID_SEARCH:
        return None
    try:
        nb_chunks = db._collection.count()
        cached = _bm25_cache.get(collection_name)
        if cached and cached[0] == nb_chunks:
            return cached[1]
        logger.info(f"BM25 : construction de l'index '{collection_name}' ({nb_chunks} chunks)…")
        result = db._collection.get(include=["documents", "metadatas"])
        texts = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not texts:
            return None
        from langchain_core.documents import Document
        docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas)]
        index = _BM25CollectionIndex(docs, texts)
        _bm25_cache[collection_name] = (nb_chunks, index)
        logger.info(f"BM25 : index prêt ({len(texts)} chunks)")
        return index
    except Exception as e:
        logger.warning(f"BM25 index indisponible ({e}) — hybrid search désactivé")
        return None


def _rrf_fusion(vector_results: list, bm25_results: list, k_final: int) -> list:
    """
    Reciprocal Rank Fusion : combine les résultats vector search et BM25.
    Chaque chunk reçoit un score RRF = Σ 1/(RRF_K + rang) pour chaque liste.
    """
    def uid(doc) -> str:
        return doc.page_content[:150]  # fingerprint unique par chunk

    scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}

    for rank, (doc, _) in enumerate(vector_results):
        u = uid(doc)
        scores[u] = scores.get(u, 0.0) + 1.0 / (RRF_K + rank + 1)
        doc_map[u] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        u = uid(doc)
        scores[u] = scores.get(u, 0.0) + 1.0 / (RRF_K + rank + 1)
        doc_map[u] = doc

    sorted_uids = sorted(scores, key=lambda u: scores[u], reverse=True)
    return [(doc_map[u], scores[u]) for u in sorted_uids[:k_final]]


# ── Tail extension (chunk N+1 début) ─────────────────────────────────────────
# Activé uniquement sur les chunks où has_continuation=True (calculé à l'indexation).
# Récupère le début du chunk N+1 via chunk_idx — sans heuristique au retrieval.
TAIL_EXTENSION_CHARS = int(os.environ.get("TAIL_EXTENSION_CHARS", "400"))


def _ajouter_tail_suivant(db, resultats: list) -> list:
    """
    Pour chaque chunk avec has_continuation=True, récupère les TAIL_EXTENSION_CHARS
    premiers caractères du chunk suivant (chunk_idx+1, même source) et les ajoute
    au contexte. Jusqu'à 2 niveaux si la continuation enchaîne.

    has_continuation est calculé à l'indexation sur le texte raw :
      True ⟺ chunk N finit par un item de liste ET chunk N+1 commence par un item de liste.
    """
    if not resultats or TAIL_EXTENSION_CHARS <= 0:
        return resultats

    from langchain_core.documents import Document

    enrichis = []
    for doc, score in resultats:
        if not doc.metadata.get("has_continuation"):
            enrichis.append((doc, score))
            continue

        source = doc.metadata.get("source")
        chunk_idx = doc.metadata.get("chunk_idx")
        if source is None or chunk_idx is None:
            enrichis.append((doc, score))
            continue

        contenu = doc.page_content
        current_idx = chunk_idx

        for niveau in range(2):  # max 2 niveaux (N+1, éventuellement N+2)
            try:
                result = db._collection.get(
                    where={"$and": [
                        {"source":    {"$eq": source}},
                        {"chunk_idx": {"$eq": current_idx + 1}},
                    ]},
                    include=["documents", "metadatas"],
                )
                next_texts = result.get("documents") or []
                next_metas = result.get("metadatas") or []
                if not next_texts:
                    break

                tail = next_texts[0][:TAIL_EXTENSION_CHARS].strip()
                if tail:
                    contenu += "\n" + tail
                    logger.debug(
                        f"Tail extension niv.{niveau + 1} : chunk_idx "
                        f"{current_idx}→{current_idx + 1} ({source})"
                    )

                # Continuer si N+1 a aussi has_continuation
                if next_metas and next_metas[0].get("has_continuation"):
                    current_idx += 1
                else:
                    break
            except Exception as e:
                logger.debug(f"Tail extension échoué : {e}")
                break

        if contenu != doc.page_content:
            enrichis.append((Document(page_content=contenu, metadata=doc.metadata), score))
        else:
            enrichis.append((doc, score))

    return enrichis


# ── Section retrieval ─────────────────────────────────────────────────────────
# Extension de section complète quand la question cible un titre spécifique
def _detecter_section_ciblee(db, question: str, resultats_initiaux: list) -> str | None:
    """
    Détecte si la question cible une section spécifique en analysant les métadonnées Docling.
    Utilise les vrais labels [TITRE-L1], [SECTION-L2] au lieu de deviner avec les majuscules.
    """
    if not resultats_initiaux or len(resultats_initiaux) > 3:
        return None

    # Analyser les métadonnées des résultats pour détecter des patterns de sections
    sources = set()
    pages = set()
    titres_detectes = []

    for doc, score in resultats_initiaux:
        sources.add(doc.metadata.get("source"))
        pages.add(doc.metadata.get("page"))

        contenu = doc.page_content

        # Chercher les labels Docling dans le contenu
        lignes = contenu.split('\n')
        for ligne in lignes[:5]:  # Premières lignes seulement
            ligne = ligne.strip()

            # Détecter les labels [TITRE-L1], [SECTION-L2], etc.
            match = re.match(r'^\[(TITRE|SECTION)-L\d+\]\s*(.+)', ligne, re.IGNORECASE)
            if match:
                label_type, titre_text = match.groups()
                titre_clean = titre_text.strip()
                if len(titre_clean) > 3 and len(titre_clean) < 100:
                    titres_detectes.append(titre_clean.lower())
                    logger.info(f"Titre Docling détecté : [{label_type}] '{titre_clean}'")

    # Si résultats dispersés (multiple sources, pages très éloignées), pas une section
    if len(sources) > 1 or (len(pages) > 1 and max(pages) - min(pages) > 3):
        return None

    # Si on trouve des titres avec labels Docling, retourner le premier
    if titres_detectes:
        titres_uniques = list(set(titres_detectes))
        if len(titres_uniques) <= 2:  # Max 2 titres différents
            titre_cible = titres_uniques[0]
            logger.info(f"Section ciblée via labels Docling : '{titre_cible}'")
            return titre_cible

    return None


def _recuperer_section_complete(db, titre_recherche: str, k_max: int = 20) -> list:
    """
    Récupère tous les chunks d'une section spécifique en cherchant le titre
    puis tous les chunks suivants jusqu'au prochain titre de même niveau.
    """
    from langchain_core.documents import Document

    if not titre_recherche:
        return []

    logger.info(f"Recherche de section complète pour : '{titre_recherche}'")

    # 1. Chercher des chunks contenant le titre (case-insensitive)
    titre_patterns = [
        titre_recherche,
        titre_recherche.upper(),
        titre_recherche.lower(),
        titre_recherche.capitalize()
    ]

    chunks_titres = []
    for pattern in titre_patterns:
        try:
            result = db._collection.get(
                where_document={"$contains": pattern},
                include=["documents", "metadatas"],
                limit=k_max
            )
            texts = result.get("documents") or []
            metas = result.get("metadatas") or []

            for text, meta in zip(texts, metas):
                if meta and text:
                    chunks_titres.append((Document(page_content=text, metadata=meta), 0.9))
        except Exception as e:
            logger.debug(f"Recherche titre '{pattern}' échouée : {e}")

    if not chunks_titres:
        return []

    # 2. Pour chaque chunk titre trouvé, récupérer sa section complète
    sections_completes = []
    sources_vues = set()

    for doc_titre, score_titre in chunks_titres:
        source = doc_titre.metadata.get("source")
        chunk_idx_titre = doc_titre.metadata.get("chunk_idx")

        if not source or chunk_idx_titre is None:
            continue

        # Éviter les doublons de source
        if source in sources_vues:
            continue
        sources_vues.add(source)

        # 3. Récupérer tous les chunks de cette source à partir du titre
        try:
            result = db._collection.get(
                where={"source": {"$eq": source}},
                include=["documents", "metadatas"],
            )
            all_texts = result.get("documents") or []
            all_metas = result.get("metadatas") or []

            # Trier par chunk_idx
            chunks_source = [(text, meta) for text, meta in zip(all_texts, all_metas) if meta]
            chunks_source.sort(key=lambda x: x[1].get("chunk_idx", 0))

            # Trouver l'index du titre dans les chunks triés
            titre_idx = None
            for i, (text, meta) in enumerate(chunks_source):
                if meta.get("chunk_idx") == chunk_idx_titre:
                    titre_idx = i
                    break

            if titre_idx is None:
                continue

            # 4. Prendre le titre + chunks suivants jusqu'au prochain titre Docling de même niveau
            section_chunks = [chunks_source[titre_idx]]  # Commencer par le titre

            # Déterminer le niveau du titre de départ
            titre_text, titre_meta = chunks_source[titre_idx]
            niveau_titre_cible = None
            first_lines = titre_text.split('\n')[:3]
            for line in first_lines:
                match = re.match(r'^\[(TITRE|SECTION)-L(\d+)\]', line.strip())
                if match:
                    niveau_titre_cible = int(match.group(2))
                    break

            for i in range(titre_idx + 1, min(len(chunks_source), titre_idx + 15)):  # Max 15 chunks par section
                text, meta = chunks_source[i]

                # Arrêter si on trouve un autre titre Docling de même niveau ou supérieur
                first_lines = text.split('\n')[:3]
                for line in first_lines:
                    match = re.match(r'^\[(TITRE|SECTION)-L(\d+)\]', line.strip())
                    if match:
                        niveau_nouveau = int(match.group(2))
                        # Arrêter si même niveau ou niveau supérieur (plus proche de la racine)
                        if niveau_titre_cible is None or niveau_nouveau <= niveau_titre_cible:
                            logger.debug(f"Arrêt section : nouveau titre niveau {niveau_nouveau} détecté")
                            break
                else:
                    # Pas de titre trouvé dans ce chunk → continuer
                    section_chunks.append((text, meta))
                    continue
                # Break trouvé dans le for interne → arrêter la section
                break

            # Convertir en format (Document, score)
            for text, meta in section_chunks:
                sections_completes.append((Document(page_content=text, metadata=meta), 0.8))

            logger.info(f"Section '{titre_recherche}' : {len(section_chunks)} chunks récupérés ({source})")

        except Exception as e:
            logger.warning(f"Erreur récupération section complète ({source}) : {e}")
            continue

    return sections_completes[:k_max]  # Limiter au cas où


# ── Keyword fallback (where_document) ─────────────────────────────────────────
# Seuil en dessous duquel le keyword fallback est déclenché.
# Reranker normalisé [0,1] : < 0.1 = aucune correspondance sémantique.
KEYWORD_FALLBACK_THRESHOLD = float(os.environ.get("KEYWORD_FALLBACK_THRESHOLD", "0.01"))

_STOPWORDS_FALLBACK = {
    # FR
    "les", "des", "pour", "dans", "avec", "sur", "par", "une", "qui", "que",
    "est", "son", "ses", "moi", "lui", "leur", "tout", "cette", "aussi",
    "donne", "mois", "references", "documents", "trouve", "trouver",
    "parle", "concernant", "cela", "ceci", "avoir", "etre", "faire",
    "peux", "mots", "toute", "base", "donnees", "infos", "informations",
    "passages", "concernant",
    # EN
    "the", "and", "for", "with", "that", "this", "from", "about",
}


def _keyword_fallback_search(db, question: str, k: int) -> list:
    """
    Recherche exacte via ChromaDB where_document lorsque la recherche sémantique
    ne trouve rien de pertinent.

    Extrait les mots significatifs de la question et cherche les chunks qui les
    contiennent littéralement (plusieurs variantes de casse tentées).
    Exemple : question="l'AMDEC" → cherche "AMDEC", "amdec", "Amdec" dans les chunks.
    """
    from langchain_core.documents import Document

    tokens = re.findall(r'\w+', question.lower())
    mots_cles = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS_FALLBACK]
    if not mots_cles:
        return []

    resultats = []
    vus: set[str] = set()

    for mot in mots_cles[:3]:
        # Variantes morphologiques : singulier ↔ pluriel + casse
        variantes_morpho = {mot}
        if mot.endswith('s') and len(mot) > 3:
            variantes_morpho.add(mot[:-1])   # directives → directive
        elif mot.endswith('aux') and len(mot) > 4:
            variantes_morpho.add(mot[:-3] + 'al')  # normaux → normal
        else:
            variantes_morpho.add(mot + 's')   # directive → directives
        # Toutes les variantes × toutes les casses
        variantes = {v for base in variantes_morpho for v in (base, base.upper(), base.capitalize())}
        for variant in variantes:
            try:
                result = db._collection.get(
                    where_document={"$contains": variant},
                    include=["documents", "metadatas"],
                )
                docs = result.get("documents") or []
                metas = result.get("metadatas") or []
                for text, meta in zip(docs, metas):
                    uid = text[:120]
                    if uid not in vus:
                        vus.add(uid)
                        resultats.append(
                            (Document(page_content=text, metadata=meta or {}), 0.5)
                        )
                        if len(resultats) >= k:
                            break
            except Exception as e:
                logger.debug(f"where_document('{variant}') échoué : {e}")
            if len(resultats) >= k:
                break
        if len(resultats) >= k:
            break

    if resultats:
        logger.info(
            f"Keyword fallback : {len(resultats)} chunk(s) trouvé(s) "
            f"pour mots-clés {mots_cles[:3]}"
        )
    return resultats


def _extraire_filtre_question(question: str) -> dict | None:
    """
    Détecte si la question cible une seule machine spécifique.
    Retourne un filtre ChromaDB si une seule machine est trouvée, None sinon.
    Exemples : "parle moi du Solo" → {"machine": {"$eq": "SOLO"}}
               "compare Solo et Gemini" → None (plusieurs machines)
    """
    q_upper = question.upper()
    machines_trouvees = [m for m in _MACHINES_CONNUES if m in q_upper]
    if len(machines_trouvees) == 1:
        return {"machine": {"$eq": machines_trouvees[0]}}
    return None


def _get_reranker():
    """
    Singleton lazy du reranker BGE.
    Le modèle est téléchargé depuis HuggingFace au premier appel (~570MB).
    Retourne None si USE_RERANKER=false ou si FlagEmbedding est absent.
    """
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance
    if not USE_RERANKER:
        return None
    try:
        from FlagEmbedding import FlagReranker
        # use_fp16=False → reranker forcé sur CPU.
        # Sur RTX 3060 Laptop 6 Go, Qwen3:8b occupe déjà ~5.0 Go GPU.
        # Le reranker sur GPU (~570 Mo fp16) dépasse le budget VRAM disponible,
        # ralentit ou fait échouer le chargement du LLM.
        # Sur CPU : cross-encoding de ~6-18 paires ≈ 1-2s de plus, imperceptible.
        # devices="cpu" : seul moyen fiable en FlagEmbedding >= 1.3 de forcer CPU.
        # use_fp16=False seul déclenche un "meta tensor" error (device_map="auto" interne).
        _reranker_instance = FlagReranker(RERANKER_MODEL, use_fp16=False, devices="cpu")
        logger.info(f"Reranker initialisé : {RERANKER_MODEL} (CPU, fp32)")
        return _reranker_instance
    except Exception as e:
        logger.warning(f"Reranker indisponible ({e}) — désactivé")
        return None


def _get_colbert():
    """
    Singleton lazy du reranker ColBERT (RAGatouille).
    Late-interaction : un vecteur par token → scoring MaxSim token-à-token.
    Meilleur que BGE sur termes techniques, acronymes, requêtes courtes.
    Modèle ~2GB téléchargé depuis HuggingFace au premier appel.
    Retourne None si USE_COLBERT=false ou si ragatouille est absent.
    """
    global _colbert_instance
    if _colbert_instance is not None:
        return _colbert_instance
    if not USE_COLBERT:
        return None
    try:
        from ragatouille import RAGPretrainedModel
        _colbert_instance = RAGPretrainedModel.from_pretrained(COLBERT_MODEL)
        logger.info(f"ColBERT initialisé : {COLBERT_MODEL}")
        return _colbert_instance
    except Exception as e:
        logger.warning(f"ColBERT indisponible ({e}) — désactivé")
        return None


def _appliquer_colbert(colbert, question: str, resultats: list, top_k: int) -> list:
    """
    Reranke les candidats via ColBERT late-interaction (RAGatouille).
    Retourne les top_k résultats triés par score décroissant.
    """
    if not resultats:
        return resultats
    try:
        docs_text = [doc.page_content for doc, _ in resultats]
        ranked = colbert.rerank(query=question, documents=docs_text, k=min(top_k, len(docs_text)))
        # ranked = [{"content": str, "score": float, "rank": int}, ...]
        content_to_original = {doc.page_content: (doc, score) for doc, score in resultats}
        reranked = []
        for item in ranked:
            original = content_to_original.get(item["content"])
            if original:
                reranked.append((original[0], float(item["score"])))
        logger.info(
            f"ColBERT : {len(resultats)} candidats → {len(reranked)} gardés "
            f"(score top : {reranked[0][1]:.3f})"
        )
        return reranked
    except Exception as e:
        logger.warning(f"ColBERT reranking échoué ({e}) — résultats originaux conservés")
        return resultats[:top_k]


def _appliquer_reranker(reranker, question: str, resultats: list, top_k: int) -> list:
    """
    Reranke les candidats ChromaDB via cross-encoder BGE.
    Retourne les top_k résultats triés par score reranker décroissant.
    Le score retourné est le score reranker normalisé (0–1, plus élevé = meilleur).
    """
    try:
        pairs = [[question, doc.page_content] for doc, _ in resultats]
        scores = reranker.compute_score(pairs, normalize=True, max_length=512)
        indexes_tries = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = [(resultats[i][0], float(scores[i])) for i in indexes_tries[:top_k]]
        logger.info(
            f"Reranker : {len(resultats)} candidats → {len(top)} gardés "
            f"(score top : {scores[indexes_tries[0]]:.3f})"
        )
        return top
    except Exception as e:
        logger.warning(f"Reranker échoué ({e}) — résultats originaux conservés")
        return resultats[:top_k]


def _reformuler_question(question: str, history: str) -> str:
    """
    Réécrit la question en query autonome et riche sémantiquement pour la recherche vectorielle.

    Exemples :
      history  = "User: parle moi de documents liés à safran\nAssistant: ..."
      question = "parle moi du plan du génie civil 2D"
      → "plan génie civil 2D offre SAFRAN équipement DED VLM Robotics"

      history  = "...SAFRAN DED cellule implantation..."
      question = "comment est donc implantée la cellule en question"
      → "implantation cellule DED métallique SAFRAN Additive Manufacturing VLM Robotics"

    Si le rewriting échoue (timeout, erreur), retourne la question originale.
    """
    if not history or history.count("User:") <= 1:
        return question

    prompt = (
        "Tu es un moteur de réécriture de requête pour un système RAG.\n"
        "À partir de l'historique de conversation et de la question actuelle, "
        "génère une requête de recherche autonome, courte et riche en mots-clés.\n"
        "La requête doit permettre de retrouver les bons documents même sans contexte.\n"
        "Réponds UNIQUEMENT avec la requête réécrite, sans explication ni ponctuation finale.\n\n"
        f"Historique :\n{history}\n\n"
        f"Question actuelle : {question}\n\n"
        "Requête de recherche :"
    )

    # Qwen3 : désactive le mode raisonnement pour le query rewriting aussi.
    # Sans /no_think, Qwen3 génère <think>...</think> qui pollue la requête vectorielle.
    if ENABLE_NO_THINK:
        prompt = "/no_think\n\n" + prompt

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # num_ctx aligné sur NUM_CTX_MIN pour éviter un rechargement du modèle
        # entre le query rewriting (ce call) et la génération principale (KvSize doit être identique).
        "options": {"temperature": 0.0, "num_ctx": NUM_CTX_MIN},
    }

    try:
        # timeout=120 : survit au chargement de Qwen3:8b (~45-50s) sans avorter.
        # Avec timeout=15, le QR abandonnait pendant le chargement → double rechargement
        # (QR annule le load en cours, puis la génération relance un 2e load de zéro).
        resp = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=120)
        resp.raise_for_status()
        rewritten = resp.json().get("response", "").strip()
        if rewritten:
            logger.info(f"Query rewriting : '{question}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewriting échoué ({e}) — question originale conservée")

    return question


def _charger_prompts_json() -> dict:
    """Charge les prompts supplémentaires depuis prompts.json s'il existe."""
    chemin = Path("prompts.json")
    if chemin.exists():
        try:
            return json.loads(chemin.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_prompt(nom: str) -> str:
    """Retourne un prompt par son nom (built-in ou depuis prompts.json)."""
    if nom in PROMPTS:
        return PROMPTS[nom]
    customs = _charger_prompts_json()
    if nom in customs:
        return customs[nom]
    return PROMPTS["defaut"]


class RAGEngine:
    """Moteur RAG : recherche de similarité + génération Ollama."""

    def __init__(self, nom_collection: str, prompt_name: str = "defaut",
                 collection_manager: CollectionManager | None = None):
        self.cm = collection_manager or CollectionManager()
        self.nom_collection = nom_collection
        self.prompt_template = get_prompt(prompt_name)
        self.db = self.cm.get_collection(nom_collection)

    def debug_retrieval(self, question: str, n_results: int = 10) -> dict:
        """
        Diagnostique la recherche ChromaDB pour une question donnée.

        Affiche pour chaque résultat : score, métadonnées, extrait du contenu.
        Retourne un dict résumé avec diagnostic et conseils.

        Usage:
            engine = RAGEngine("vlm_robotics", prompt_name="vlm_robotics")
            summary = engine.debug_retrieval("Liste les projets SOLO")
        """
        RESET = "\033[0m"
        BOLD = "\033[1m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        CYAN = "\033[96m"

        def score_label(s: float) -> str:
            # Seuils calibrés pour distance L2 (retournée par LangChain+ChromaDB).
            # Équivalences cosine : L2=0.63 → cos_sim=80%, L2=0.85 → cos_sim=64%
            # Formule : cos_sim = 1 - L2² / 2  (vecteurs normalisés)
            if s < 0.45:
                return f"{GREEN}EXCELLENT ({s:.4f}){RESET}"
            if s < 0.63:
                return f"{CYAN}BON ({s:.4f}){RESET}"
            if s < 0.85:
                return f"{YELLOW}MOYEN ({s:.4f}){RESET}"
            return f"{RED}FAIBLE ({s:.4f}){RESET}"

        print(f"\n{BOLD}{'═'*65}{RESET}")
        print(f"{BOLD}  DEBUG RETRIEVAL — collection : {self.nom_collection}{RESET}")
        print(f"{BOLD}{'═'*65}{RESET}")
        print(f"  Question  : {question}")
        print(f"  n_results : {n_results}")
        print(f"{'─'*65}\n")

        resultats = self.db.similarity_search_with_score(question, k=n_results)
        nb = len(resultats)
        print(f"  {BOLD}Résultats trouvés : {nb}{RESET}\n")

        scores: list[float] = []
        items: list[dict] = []

        for i, (doc, score) in enumerate(resultats, 1):
            scores.append(score)
            source = doc.metadata.get("source", "Inconnu")
            page = doc.metadata.get("page", "?")
            extrait = doc.page_content[:200].replace("\n", " ")

            print(f"  [{i}] {score_label(score)}")
            print(f"       Source   : {source}  (page {page})")
            print(f"       Métadata : {doc.metadata}")
            print(f"       Contenu  : {extrait}…")
            print()

            items.append({
                "rang": i,
                "score": score,
                "source": source,
                "page": page,
                "metadata": doc.metadata,
                "extrait": doc.page_content[:200],
            })

        # ── Diagnostic global ───────────────────────────────────────────
        # Seuils L2 (ChromaDB retourne distance L2, pas cosine)
        # L2 < 0.45 → cos_sim > 90% | L2 < 0.63 → cos_sim > 80% | L2 > 0.85 → cos_sim < 64%
        if nb == 0:
            diag = "AUCUN RÉSULTAT — problème de données ou de collection"
            conseil = (
                "Vérifiez que des documents sont bien indexés :\n"
                "  python ingest.py vlm_robotics ./documents/"
            )
        elif min(scores) > 0.85:
            diag = "SCORES TRÈS ÉLEVÉS — problème d'embedding probable"
            conseil = (
                "Le modèle d'embedding utilisé à la requête diffère probablement\n"
                "  de celui utilisé à l'indexation.\n"
                "  → Vérifiez EMBEDDING_MODEL dans core/embeddings.py\n"
                "  → Réindexez : python ingest.py vlm_robotics ./documents/ --force"
            )
        elif min(scores) > 0.63:
            diag = "SCORES ÉLEVÉS — mismatch sémantique ou chunking trop grand"
            conseil = (
                "Les chunks ne correspondent pas bien à la requête.\n"
                "  → Reformulez avec les termes exacts du document\n"
                "  → Réduisez chunk_size (1000 → 500) dans document_manager.py\n"
                "  → Utilisez un filtre metadata si le client/machine est connu"
            )
        else:
            diag = "SCORES BONS — retrieval fonctionnel"
            conseil = (
                "La recherche vectorielle fonctionne (cos_sim > 80%).\n"
                "  Si la réponse LLM est mauvaise, vérifiez le prompt\n"
                "  ou augmentez K (NB_CHUNKS_RECHERCHE dans search.py)."
            )

        score_min = min(scores) if scores else None
        score_moy = sum(scores) / len(scores) if scores else None
        score_max = max(scores) if scores else None

        print(f"{'─'*65}")
        print(f"  {BOLD}DIAGNOSTIC : {diag}{RESET}")
        print(f"  Conseil    : {conseil}")
        if score_min is not None:
            print(f"\n  Score min : {score_min:.4f}  |  moy : {score_moy:.4f}  |  max : {score_max:.4f}")
        print(f"{BOLD}{'═'*65}{RESET}\n")

        return {
            "question": question,
            "nb_resultats": nb,
            "score_min": score_min,
            "score_moy": score_moy,
            "score_max": score_max,
            "scores": scores,
            "diagnostic": diag,
            "conseil": conseil,
            "resultats": items,
        }

    def _adapter_parametres(self) -> tuple[int, int]:
        """
        Calcule K et num_ctx dynamiquement selon la taille réelle de la collection.

        Règles :
          K       = max(6, min(nb_chunks // 8, 20))
          num_ctx = K × ~250 tokens/chunk + 1200 overhead, borné entre 4096 et 32768

        Exemples :
          48  chunks → K=6,  num_ctx=4096
          446 chunks → K=20, num_ctx=6200 (→ 8192 par l'arrondi)
        """
        try:
            nb_chunks = self.db._collection.count()
        except Exception:
            nb_chunks = 0

        if nb_chunks == 0:
            return NB_CHUNKS_RECHERCHE, NUM_CTX_MIN

        k = max(6, min(nb_chunks // 8, 20))
        tokens_par_chunk = CHUNK_SIZE_APPROX // 4   # ~250 tokens
        overhead = 1200                              # prompt + historique + question
        num_ctx = k * tokens_par_chunk + overhead
        num_ctx = max(NUM_CTX_MIN, min(num_ctx, NUM_CTX_MAX))
        return k, num_ctx

    def rechercher(self, question: str, k: int = NB_CHUNKS_RECHERCHE,
                   history: str = "") -> tuple[str, list[dict]]:
        """
        Recherche les chunks les plus pertinents.

        Pipeline :
          1. Détecte si la question cible une machine spécifique → filtre metadata
          2. Récupère k×3 candidats (si reranker actif) ou k directement
          3. Recherche vectorielle ChromaDB
          4. Hybrid BM25 : fusionne vector + BM25 via RRF → k_candidats meilleurs
          5. Reranke via cross-encoder BGE → garde les top k
          6. Seuil relatif : écarte les chunks au score < score_max × 0.1
          7. Déduplication PDF/DOCX : élimine les doublons entre formats
          8. Retourne (contexte_texte, liste_sources)
        """
        # 1. Pré-filtrage metadata
        filtre = _extraire_filtre_question(question)
        if filtre:
            logger.info(f"Filtre metadata : {filtre}")

        # 2. Nombre de candidats à récupérer
        reranker = _get_reranker()
        k_candidats = min(k * RERANKER_CANDIDATS_MULT, RERANKER_CANDIDATS_MAX) if reranker else k

        # 3. Recherche vectorielle
        try:
            resultats = self.db.similarity_search_with_score(question, k=k_candidats, filter=filtre)
            if not resultats and filtre:
                logger.info(f"Aucun résultat avec filtre {filtre}, retry sans filtre")
                resultats = self.db.similarity_search_with_score(question, k=k_candidats)
        except Exception as e:
            logger.warning(f"Recherche avec filtre échouée ({e}) — retry sans filtre")
            resultats = self.db.similarity_search_with_score(question, k=k_candidats)

        # 4. Hybrid BM25 — fusion avec RRF sur les mêmes k_candidats
        bm25_index = _get_or_build_bm25(self.db, self.nom_collection)
        if bm25_index:
            bm25_results = bm25_index.search(question, k=k_candidats, filtre=filtre)
            resultats = _rrf_fusion(resultats, bm25_results, k_final=k_candidats)

        # 5. Reranking — ColBERT (prioritaire si USE_COLBERT=true) ou BGE
        colbert = _get_colbert()
        if colbert and len(resultats) > k:
            resultats = _appliquer_colbert(colbert, question, resultats, top_k=k)
        elif reranker and resultats:
            resultats = _appliquer_reranker(reranker, question, resultats, top_k=k)

        # 5b. Keyword fallback — si le reranker (ou la RRF sans reranker) ne trouve
        # rien de pertinent, recherche exacte par mots-clés via where_document.
        score_best = resultats[0][1] if resultats else 0.0
        if score_best < KEYWORD_FALLBACK_THRESHOLD:
            fallback = _keyword_fallback_search(self.db, question, k=k)
            if fallback:
                # On prepend les résultats keyword : ils ont une correspondance exacte
                resultats = fallback + [r for r in resultats if r not in fallback]

        # 6. Seuil reranker relatif : élimine les chunks trop éloignés du meilleur
        # (score < score_max × 0.1). Le meilleur chunk passe toujours ce seuil.
        if reranker and resultats:
            score_max_r = resultats[0][1]  # résultats déjà triés par score décroissant
            seuil_relatif = score_max_r * 0.1
            nb_avant = len(resultats)
            resultats = [(doc, s) for doc, s in resultats if s >= seuil_relatif]
            nb_filtres = nb_avant - len(resultats)
            if nb_filtres:
                logger.info(
                    f"Seuil relatif reranker (≥{seuil_relatif:.3f}) : "
                    f"{nb_filtres} chunk(s) écarté(s) sur {nb_avant}"
                )

        # 6b. Seuil absolu : élimine les chunks quasi-nuls même après keyword fallback.
        # Quand aucune recherche ne trouve rien, on retourne [] → le LLM dira
        # "je n'ai pas trouvé" plutôt que de servir du contexte non pertinent.
        # Seuil 0.01 intentionnellement bas : keyword fallback = 0.5, reranker OK > 0.1.
        nb_avant_absolu = len(resultats)
        resultats = [(doc, s) for doc, s in resultats if s >= 0.01]
        if len(resultats) < nb_avant_absolu:
            logger.info(
                f"Seuil absolu (≥0.01) : {nb_avant_absolu - len(resultats)} "
                f"chunk(s) quasi-nuls éliminés"
            )

        # 7. Déduplication PDF/DOCX
        resultats = _deduplicater_pdf_docx(resultats)

        # 7b. Tail extension — complète les chunks qui s'arrêtent au milieu d'une liste
        resultats = _ajouter_tail_suivant(self.db, resultats)

        # 7c. Section complète — détection intelligente via analyse des résultats initiaux
        titre_cible = _detecter_section_ciblee(self.db, question, resultats)
        if titre_cible:
            logger.info(f"Section ciblée détectée : '{titre_cible}' → recherche section complète")
            section_complete = _recuperer_section_complete(self.db, titre_cible, k_max=k*2)
            if section_complete and len(section_complete) > len(resultats):
                # Remplacer seulement si on a plus de contenu
                resultats = section_complete
                logger.info(f"Section complète récupérée : {len(section_complete)} chunks (remplace {len(resultats)} résultats)")

        # 8. Formater les résultats
        contexte_parts = []
        sources = []
        sources_vues = set()

        for doc, score in resultats:
            content = doc.page_content
            # Le préfixe contextuel "[résumé]\n\n" est ajouté à l'indexation pour enrichir
            # l'embedding. On le déplace en fin de chunk pour que le LLM lise le contenu
            # complet en premier (sinon il répond avec le résumé au lieu du détail).
            m = re.match(r'^(\[[^\n\]]+\])\n\n(.+)', content, re.DOTALL)
            if m:
                content = m.group(2) + "\n\n" + m.group(1)
            contexte_parts.append(content)
            cle_source = f"{doc.metadata.get('source', 'Inconnu')} - p.{doc.metadata.get('page', '?')}"
            if cle_source not in sources_vues:
                sources_vues.add(cle_source)
                sources.append({
                    "fichier": doc.metadata.get("source", "Inconnu"),
                    "page": doc.metadata.get("page", "?"),
                    "score": round(score, 3),
                })

        contexte = "\n\n---\n\n".join(contexte_parts)
        return contexte, sources

    def rechercher_debug(self, question: str, history: str = "") -> dict:
        """
        Retourne les données brutes du pipeline de retrieval pour debug frontend.

        Retourne :
          {
            "original_question": str,
            "rewritten_query": str,
            "chunks": [
              {
                "rank": int,
                "score": float,
                "source": str,
                "page": str|int,
                "chunk_idx": int,
                "machine": str,
                "sections": list[str],   # hierarchy_parents désérialisé
                "content": str,          # contenu complet
                "content_preview": str,  # 300 premiers chars
              }
            ]
          }
        """
        k, _ = self._adapter_parametres()
        history_text = history if history else ""

        rewritten_query = _reformuler_question(question, history_text)
        _, sources_list = self.rechercher(rewritten_query, k=k, history=history_text)

        # Re-run pour récupérer les docs complets (rechercher() retourne contexte str + sources)
        # On refait le pipeline directement ici pour avoir les docs avec contenu
        filtre = _extraire_filtre_question(rewritten_query)
        reranker = _get_reranker()
        k_candidats = min(k * RERANKER_CANDIDATS_MULT, RERANKER_CANDIDATS_MAX) if reranker else k

        try:
            resultats = self.db.similarity_search_with_score(rewritten_query, k=k_candidats, filter=filtre)
            if not resultats and filtre:
                resultats = self.db.similarity_search_with_score(rewritten_query, k=k_candidats)
        except Exception:
            resultats = self.db.similarity_search_with_score(rewritten_query, k=k_candidats)

        bm25_index = _get_or_build_bm25(self.db, self.nom_collection)
        if bm25_index:
            bm25_results = bm25_index.search(rewritten_query, k=k_candidats, filtre=filtre)
            resultats = _rrf_fusion(resultats, bm25_results, k_final=k_candidats)

        colbert = _get_colbert()
        if colbert and len(resultats) > k:
            resultats = _appliquer_colbert(colbert, rewritten_query, resultats, top_k=k)
        elif reranker and resultats:
            resultats = _appliquer_reranker(reranker, rewritten_query, resultats, top_k=k)

        resultats = _deduplicater_pdf_docx(resultats)
        resultats = _ajouter_tail_suivant(self.db, resultats)

        chunks = []
        for rank, (doc, score) in enumerate(resultats, 1):
            meta = doc.metadata or {}
            # Désérialiser hierarchy_parents (stocké en JSON string)
            sections_raw = meta.get("hierarchy_parents", "[]")
            try:
                sections = json.loads(sections_raw) if isinstance(sections_raw, str) else sections_raw
            except Exception:
                sections = []

            chunks.append({
                "rank": rank,
                "score": round(score, 4),
                "source": meta.get("source", "?"),
                "page": meta.get("page", "?"),
                "chunk_idx": meta.get("chunk_idx"),
                "machine": meta.get("machine"),
                "sections": sections,
                "content": doc.page_content,
                "content_preview": doc.page_content[:300],
            })

        return {
            "original_question": question,
            "rewritten_query": rewritten_query,
            "chunks": chunks,
        }

    def generer_avec_sources(self, question: str, stream: bool = True, history: str = "") -> dict:
        """
        Recherche + génération LLM.

        Retourne {"reponse": generator|str, "sources": list[dict]}
        """
        k, num_ctx = self._adapter_parametres()

        # Réécrire la question en query autonome si un historique est disponible
        query_recherche = _reformuler_question(question, history)

        contexte, sources = self.rechercher(query_recherche, k=k, history=history)

        # Ajouter l'historique au contexte si fourni
        if history:
            contexte = f"Historique de conversation:\n{history}\n\n---\n\n{contexte}"

        prompt = self.prompt_template.format(context=contexte, question=question)

        reponse = self._appeler_ollama(prompt, stream=stream, num_ctx=num_ctx)
        return {"reponse": reponse, "sources": sources}

    @staticmethod
    def _appeler_ollama(prompt: str, stream: bool = True, num_ctx: int = NUM_CTX_MIN):
        """
        Appelle l'API Ollama.
        Si stream=True, retourne un générateur de tokens.
        Si stream=False, retourne la réponse complète (str).
        num_ctx est calculé dynamiquement par _adapter_parametres().
        """
        if ENABLE_NO_THINK:
            prompt = "/no_think\n\n" + prompt

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": "30m",  # Garde le modèle en mémoire 30 minutes
            "options": {
                "temperature": 0.3,
                "num_ctx": num_ctx,
            },
        }

        try:
            reponse = requests.post(
                OLLAMA_API_GENERATE,
                json=payload,
                stream=stream,
                timeout=600,  # 10 minutes au lieu de 5
            )
            reponse.raise_for_status()
        except requests.ConnectionError:
            msg = "Impossible de contacter Ollama. Vérifiez qu'il est lancé avec `ollama serve`."
            if stream:
                def _err():
                    yield msg
                return _err()
            return msg
        except requests.Timeout:
            msg = "Ollama n'a pas répondu à temps. Réessayez."
            if stream:
                def _err():
                    yield msg
                return _err()
            return msg
        except requests.HTTPError as e:
            msg = f"Erreur Ollama : {e}"
            if stream:
                def _err():
                    yield msg
                return _err()
            return msg

        if not stream:
            data = reponse.json()
            return data.get("response", "")

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
            except requests.exceptions.ChunkedEncodingError as e:
                logger.warning(f"Stream interrompu : {e}")
            finally:
                reponse.close()

        return _stream_tokens()
