"""
core/search.py — RAGEngine : recherche similarité + génération Ollama streaming.
"""

import hashlib
import json
import logging
import os
import re
import time
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
- Cite la source quand elle est disponible (nom du fichier, section, page si précisée dans le contexte).
- Réponds directement à la question posée en t'appuyant sur le contexte fourni.
- Ne recommande une machine spécifique QUE si la question le demande explicitement.
- Si l'information n'est pas dans le contexte fourni, dis-le clairement : « Je n'ai pas trouvé cette information dans la documentation disponible. »
- Ne jamais inventer de spécifications techniques.
- Reproduis les listes du contexte de manière exhaustive : cite TOUS les éléments sans en omettre, abréger ou résumer aucun.
- N'ajoute jamais d'explications ou de phrases qui ne sont pas explicitement dans le contexte fourni.
- Les préfixes `[Section: ...]` dans le contexte sont des métadonnées internes de navigation — ne les reproduis jamais dans ta réponse."""

# Prompts nommés disponibles
PROMPTS = {
    "defaut": PROMPT_DEFAUT,
    "vlm_robotics": PROMPT_VLM_ROBOTICS,
    "vlm": PROMPT_VLM_ROBOTICS,  # alias : collection "vlm" → prompt VLM Robotics
}

# ── Contexte domaine pour HyDE ────────────────────────────────────────────────
# Descriptions courtes (~50 tokens) injectées dans _generer_document_hypothetique()
# pour guider le LLM vers le bon vocabulaire technique avant la retrieval.
# TODO: enrichir avec metadata 'technology' (WAAM, laser, Cold Spray, FSW...)
#       si indexé dans les documents → contexte encore plus précis par procédé
# TODO: enrichir avec metadata 'sector' (ASD, Naval, Ferroviaire...)
#       pour orienter le document hypothétique vers le bon secteur client
CONTEXTE_DOMAINE_MACHINE: dict[str, str] = {
    "SOLO":    "SOLO : machine hybride XXL mono-robot VLM Robotics. Procédés : WAAM (CMT Fronius), usinage, CND, scan. Grandes pièces structurelles métalliques.",
    "GEMINI":  "GEMINI : machine hybride XXL bi-robot VLM Robotics, la plus avancée. Multi-procédés simultanés, haute productivité.",
    "COMPAQT": "COMPAQT XL : machine hybride entrée de gamme VLM Robotics. Fabrication additive (WAAM/laser) + usinage intégrés.",
    "HYMANCO": "HYMANCO : unité mobile containerisée VLM Robotics. Interventions terrain MRO, déployable sur site.",
}
CONTEXTE_VLM_DEFAUT = (
    "VLM Robotics : constructeur de machines-outils hybrides XXL (fabrication additive + usinage). "
    "Gamme : COMPAQT, SOLO, GEMINI, HYMANCO. Technologies : WAAM, laser, Cold Spray, FSW, CND."
)

NB_CHUNKS_RECHERCHE = 6   # fallback si collection vide
CHUNK_SIZE_APPROX = 1000  # doit correspondre à document_manager.CHUNK_SIZE
NUM_CTX_MIN = 4096
# Plafond hardware pour RTX 3060 6GB + qwen3.5:4b :
#   weights GPU 3.1GB + KV q8_0 @8192 ~1.7GB + compute 0.76GB = ~5.6GB ✅
#   KV q8_0 @12288 ~2.5GB → total ~6.4GB → OOM
# Mettre OLLAMA_NUM_CTX_HARD_MAX=32768 sur RTX 5090.
NUM_CTX_HARD_MAX = int(os.environ.get("OLLAMA_NUM_CTX_HARD_MAX", "8192"))
NUM_CTX_MAX = 32768  # plafond absolu pour _adapter_parametres (borné par HARD_MAX ensuite)

# Brackets fixes pour éviter les rechargements Ollama entre calls successifs.
# Quand num_ctx change entre query-rewriting (4096) et génération (autre valeur),
# Ollama décharge + recharge le modèle (~12-25s de latence).
# On arrondit au bracket le plus proche ≤ HARD_MAX : les calls qui tombent dans
# le même bracket ne déclenchent pas de rechargement.
# RTX 3060 (HARD_MAX=8192)  : brackets actifs = [4096, 6200, 8192]
# RTX 5090 (HARD_MAX=32768) : brackets actifs = [4096, 6200, 8192, 12288, 16384, 32768]
_NUM_CTX_BRACKETS = [4096, 6200, 8192, 12288, 16384, 32768]


def _bracket_num_ctx(tokens_estimes: int) -> int:
    """
    Arrondit au bracket fixe le plus proche pour éviter les rechargements Ollama.
    Ne dépasse jamais NUM_CTX_HARD_MAX (limite hardware configurée).
    Si aucun bracket ne suffit, retourne HARD_MAX directement.
    """
    for b in _NUM_CTX_BRACKETS:
        if tokens_estimes <= b and b <= NUM_CTX_HARD_MAX:
            return b
    return NUM_CTX_HARD_MAX

# ── Metadata filtering ────────────────────────────────────────────────────────
_MACHINES_CONNUES = ["GEMINI", "SOLO", "COMPAQT", "HYMANCO"]

# ── Reranker config ───────────────────────────────────────────────────────────
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER_GPU = os.environ.get("USE_RERANKER_GPU", "false").lower() == "true"
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

# ── Group A — nouvelles fonctionnalités RAG ───────────────────────────────────
MAX_CHUNKS_PER_SOURCE = int(os.environ.get("MAX_CHUNKS_PER_SOURCE", "0"))  # 0=désactivé
USE_MULTI_QUERY = os.environ.get("USE_MULTI_QUERY", "true").lower() == "true"
USE_HYDE = os.environ.get("USE_HYDE", "false").lower() == "true"
USE_MMR = os.environ.get("USE_MMR", "true").lower() == "true"
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.6"))
USE_CRAG = os.environ.get("USE_CRAG", "false").lower() == "true"
CRAG_QUALITY_THRESHOLD = float(os.environ.get("CRAG_QUALITY_THRESHOLD", "0.5"))
FILTER_THINK_FROM_STREAM = os.environ.get("FILTER_THINK_FROM_STREAM", "false").lower() == "true"

# ── Group B — Qdrant + bge-m3 + parent/child ─────────────────────────────────
# VECTOR_DB=qdrant : désactive BM25, keyword_fallback, tail_extension, section_retrieval
#   (ces features utilisent db._collection — API ChromaDB non disponible avec Qdrant)
# USE_PARENT_CHILD=true : remplace tail_extension et section_retrieval par parent_text
#   Le LLM reçoit le parent (800-1200 tokens) au lieu du child (250 tokens)
VECTOR_DB = os.environ.get("VECTOR_DB", "chroma").lower()
USE_PARENT_CHILD = os.environ.get("USE_PARENT_CHILD", "false").lower() == "true"


def build_search_config_hash() -> str:
    """
    Hash SHA256[:8] des paramètres de recherche (query-time).
    Change sans re-indexation → déclenche un nouveau groupe dans le dashboard eval.
    """
    config = {
        "hyde": USE_HYDE,
        "crag": USE_CRAG,
        "multi_query": USE_MULTI_QUERY,
        "mmr": USE_MMR,
        "reranker": USE_RERANKER,
        "reranker_model": RERANKER_MODEL if USE_RERANKER else None,
        "parent_child": USE_PARENT_CHILD,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def _get_pipeline_hash() -> str:
    """Retourne le hash de la pipeline d'indexation (depuis document_manager)."""
    try:
        from core.document_manager import build_pipeline_fingerprint
        return build_pipeline_fingerprint()["hash"]
    except Exception:
        return "unknown"


_RE_IMAGE_LABEL = re.compile(r'\[IMAGE(?:-L\d+)?\][^\n]*', re.IGNORECASE)

def _filtrer_chunks_image_seule(resultats: list, seuil_chars: int = 80) -> list:
    """
    Filtre les chunks dont le contenu utile (hors labels [IMAGE...]) est trop court.

    Un chunk "image seule" = titre de section + image sans texte extracté.
    Ces chunks polluent le contexte LLM et le reranker avec du contenu non-textuel.

    TODO: désactiver ce filtre quand l'OCR sera intégré (Docling VLM / Tesseract).
          À ce moment, les images auront du texte extracté et le [IMAGE] label
          sera complété par le contenu visuel reconnu → chunks utiles à conserver.
    """
    filtres = []
    nb_exclus = 0
    for doc, score in resultats:
        texte_sans_image = _RE_IMAGE_LABEL.sub("", doc.page_content).strip()
        if len(texte_sans_image) < seuil_chars:
            nb_exclus += 1
            continue
        filtres.append((doc, score))
    if nb_exclus:
        logger.info(f"Filtre image-seule : {nb_exclus} chunk(s) exclus (< {seuil_chars} chars utiles)")
    return filtres


def _cap_par_source(resultats: list, max_per_source: int) -> list:
    """
    A2 — Limite le nombre de chunks par source document.
    Désactivé si max_per_source=0 (MAX_CHUNKS_PER_SOURCE=0).
    Appliqué après reranking pour ne capper que les chunks les plus pertinents.
    Valeur recommandée quand activé : 5 (permet les longues explications).
    """
    if max_per_source <= 0:
        return resultats
    counts: dict[str, int] = {}
    filtered = []
    for doc, score in resultats:
        source = doc.metadata.get("source", "")
        base = re.sub(r'\.(pdf|docx|doc|txt)$', '', source, flags=re.IGNORECASE)
        count = counts.get(base, 0)
        if count < max_per_source:
            filtered.append((doc, score))
            counts[base] = count + 1
    nb_removed = len(resultats) - len(filtered)
    if nb_removed:
        logger.info(f"Cap par source (max={max_per_source}) : {nb_removed} chunk(s) écarté(s)")
    return filtered


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
    seen: dict[tuple, tuple[float, int]] = {}  # (base, page, chunk_idx) → (score, index)

    for i, (doc, score) in enumerate(resultats):
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "?")
        chunk_idx = doc.metadata.get("chunk_idx", i)
        base = re.sub(r'\.(pdf|docx|doc|txt)$', '', source, flags=re.IGNORECASE)
        # chunk_idx dans la clé : PDF vs DOCX du même doc ont le même chunk_idx → dédupliqués
        # Deux chunks différents de la même page ont des chunk_idx différents → conservés
        key = (base, page, chunk_idx)

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
    Désactivé si VECTOR_DB=qdrant (Qdrant HYBRID natif remplace BM25).
    """
    if not USE_HYBRID_SEARCH or VECTOR_DB == "qdrant":
        return None
    if not hasattr(db, "_collection"):
        return None  # Qdrant store sans _collection — BM25 indisponible
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


def _rrf_fusion_lists(result_lists: list, k_final: int) -> list:
    """
    A5 — RRF fusion sur plusieurs listes de résultats (multi-query).
    Chaque liste reçoit un score RRF = Σ 1/(RRF_K + rang) par chunk présent.
    """
    def uid(doc) -> str:
        return doc.page_content[:150]

    scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}

    for result_list in result_lists:
        for rank, (doc, _) in enumerate(result_list):
            u = uid(doc)
            scores[u] = scores.get(u, 0.0) + 1.0 / (RRF_K + rank + 1)
            doc_map[u] = doc

    sorted_uids = sorted(scores, key=lambda u: scores[u], reverse=True)
    return [(doc_map[u], scores[u]) for u in sorted_uids[:k_final]]


def _generer_variantes_question(question: str, history: str, hyde_doc: str = "") -> list:
    """
    A5 — Génère 2 variantes sémantiques de la question pour le multi-query retrieval.
    - hyde_doc : si fourni (HyDE actif), inspire les variantes avec le vocabulaire technique
                 du document hypothétique → variantes plus ciblées que la question brute
    Retourne [original, variante1, variante2] ou [original] si échec.
    """
    if not USE_MULTI_QUERY:
        return [question]

    contexte_hyde = ""
    if hyde_doc and hyde_doc != question:
        # Tronquer le doc HyDE pour rester dans num_ctx=4096 (~200 tokens max)
        contexte_hyde = (
            f"Contexte technique (vocabulaire à utiliser dans les variantes) :\n"
            f"{hyde_doc[:800]}\n\n"
        )

    prompt = (
        "Tu es un moteur de diversification de requêtes pour un système RAG.\n"
        f"{contexte_hyde}"
        "Génère 2 reformulations différentes de la question ci-dessous, "
        "couvrant des aspects ou formulations complémentaires.\n"
        "Réponds UNIQUEMENT avec les 2 variantes, une par ligne, sans numérotation ni explication.\n\n"
        f"Question : {question}\n\n"
        "Variantes :"
    )
    if ENABLE_NO_THINK:
        prompt = "/no_think\n\n" + prompt

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_ctx": NUM_CTX_HARD_MAX},  # aligné sur tool_loop — évite reload KV cache
    }
    try:
        resp = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=500)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        lignes = [l.strip() for l in text.split('\n') if l.strip()]
        variantes = lignes[:2]
        if variantes:
            result = [question] + variantes
            logger.info(f"Multi-query : {len(result)} variantes générées")
            for i, v in enumerate(result):
                logger.info(f"  Query #{i+1} : {v}")
            return result
    except Exception as e:
        logger.warning(f"Multi-query variantes échouées ({e}) — query originale seulement")
    return [question]


def _generer_document_hypothetique(
    question: str,
    history: str = "",
    filtre: dict | None = None,
    nom_collection: str = "",
    hyde_mode: str = "narrative",
) -> str:
    """
    A6 — HyDE (Hypothetical Document Embeddings).
    Génère un document hypothétique qui répondrait à la question.
    - filtre         : metadata filter (ex: machine=SOLO) — priorité haute pour contexte machine
    - nom_collection : nom de la collection — fallback contexte VLM général si pas de machine
    - history        : historique de conv pour les warm turns
    - hyde_mode      : "narrative" (défaut, paragraphes descriptifs) ou "composition"
                       (liste de références produits — pour le RFQPlanner)
    Retourne la question originale si échec.
    """
    # Construire le contexte domaine — priorité : machine spécifique > VLM général > rien
    # Permet à HyDE de générer un document hypothétique pertinent même sans historique (cold start).
    contexte_domaine = ""
    if filtre:
        for cle, val in filtre.items():
            if isinstance(val, dict) and "$eq" in val:
                machine = str(val["$eq"]).upper()
                if machine in CONTEXTE_DOMAINE_MACHINE:
                    contexte_domaine = f"Contexte : {CONTEXTE_DOMAINE_MACHINE[machine]}\n"
                    break
    if not contexte_domaine and "vlm" in nom_collection.lower():
        contexte_domaine = f"Contexte : {CONTEXTE_VLM_DEFAUT}\n"
    # TODO: si metadata 'technology' disponible dans filtre → injecter description procédé
    # TODO: si metadata 'sector' disponible dans filtre → injecter contexte secteur client

    # Construire le contexte historique (warm turns) — uniquement en mode narrative
    contexte_history = ""
    if history and hyde_mode == "narrative":
        contexte_history = f"Historique de conversation (termes et contexte domaine) :\n{history}\n\n"

    if hyde_mode == "composition":
        # Mode RFQPlanner : génère une liste de composants SPÉCIFIQUES à la dimension demandée.
        # IMPORTANT : ne pas générer une BOM complète machine → pollution du multi-query avec
        # termes hors-scope (ex: chercher "Source Laser" ne doit pas retourner Comau/VLMV3T).
        # Chaque dimension = sa propre liste courte de composants DIRECTS uniquement.
        prompt = (
            "Tu es un générateur de listes de composants pour un système RAG industriel.\n"
            f"{contexte_domaine}"
            "Génère une liste courte (5-8 lignes) de composants techniques, références produits "
            "et marques SPÉCIFIQUES à la dimension technique suivante UNIQUEMENT. "
            "Ne liste QUE les composants directement liés à cette dimension précise "
            "(ex: pour 'Source Laser' → uniquement lasers, têtes laser, optiques, fibres optiques ; "
            "pour 'CN / Pupitre' → uniquement automates, CNC, IHM, écrans opérateur ; "
            "pour 'Robot' → uniquement bras robotiques, contrôleurs robot). "
            "Ne pas inclure des composants d'autres dimensions. "
            "Utilise des noms de produits réels et précis. "
            "Format : une référence par ligne, nom du composant suivi de sa marque/modèle.\n"
            "Réponds UNIQUEMENT avec la liste, sans introduction ni explication.\n\n"
            f"Dimension technique à détailler : {question}\n\n"
            "Liste de composants spécifiques à cette dimension :"
        )
    else:
        prompt = (
            "Tu es un générateur de documents hypothétiques pour un système RAG.\n"
            f"{contexte_domaine}"
            "Génère 1-2 paragraphes de documentation technique qui répondraient "
            "à la question suivante. Utilise le vocabulaire technique des documents sources.\n"
            "Réponds UNIQUEMENT avec le document hypothétique, sans introduction ni explication.\n\n"
            f"{contexte_history}"
            f"Question : {question}\n\n"
            "Document hypothétique :"
        )
    if ENABLE_NO_THINK:
        prompt = "/no_think\n\n" + prompt

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.5, "num_ctx": NUM_CTX_HARD_MAX},  # aligné sur tool_loop — évite reload KV cache
    }
    try:
        resp = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=500)
        resp.raise_for_status()
        doc = resp.json().get("response", "").strip()
        if doc:
            logger.info(f"HyDE : document hypothétique généré ({len(doc)} chars)\n--- HYDE DOC ---\n{doc}\n--- FIN HYDE ---")
            return doc
    except Exception as e:
        logger.warning(f"HyDE échoué ({e}) — question originale conservée")
    return question


def _appliquer_mmr(resultats: list, lambda_mult: float = MMR_LAMBDA, k: int = 0) -> list:
    """
    A4 — Maximal Marginal Relevance : diversifie les résultats tout en gardant la pertinence.
    lambda_mult=0.6 → 60% relevance, 40% diversity.
    Algorithme greedy : sélectionne le chunk maximisant λ×relevance − (1−λ)×max_sim_to_selected.
    """
    if not USE_MMR or len(resultats) <= 1:
        return resultats

    target_k = k if k > 0 else len(resultats)

    try:
        import numpy as np
        from core.embeddings import get_embeddings

        texts = [doc.page_content for doc, _ in resultats]
        scores = [score for _, score in resultats]
        score_max = max(scores) if scores else 1.0
        norm_scores = [s / score_max if score_max > 0 else s for s in scores]

        embs = get_embeddings().embed_documents(texts)
        embs_np = np.array(embs, dtype=float)
        norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embs_norm = embs_np / norms

        selected_indices: list[int] = []
        remaining = list(range(len(resultats)))

        while remaining and len(selected_indices) < target_k:
            if not selected_indices:
                best_idx = max(remaining, key=lambda i: norm_scores[i])
            else:
                selected_embs = embs_norm[selected_indices]
                best_mmr = -float("inf")
                best_idx = remaining[0]
                for i in remaining:
                    relevance = norm_scores[i]
                    sims = embs_norm[i] @ selected_embs.T
                    max_sim = float(np.max(sims))
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_idx = i
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        result = [resultats[i] for i in selected_indices]
        logger.info(f"MMR : {len(resultats)} chunks → {len(result)} retenus (λ={lambda_mult})")
        return result

    except Exception as e:
        logger.warning(f"MMR échoué ({e}) — résultats originaux conservés")
        return resultats


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

    Désactivé si :
      - USE_PARENT_CHILD=true : le parent_text fournit déjà le contexte élargi
      - VECTOR_DB=qdrant : db._collection non disponible
    """
    if not resultats or TAIL_EXTENSION_CHARS <= 0:
        return resultats
    if USE_PARENT_CHILD or VECTOR_DB == "qdrant" or not hasattr(db, "_collection"):
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
    Désactivé si USE_PARENT_CHILD=true ou VECTOR_DB=qdrant (db._collection non disponible).
    """
    if USE_PARENT_CHILD or VECTOR_DB == "qdrant" or not hasattr(db, "_collection"):
        return None
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
    # FR — articles, pronoms, prépositions
    "les", "des", "pour", "dans", "avec", "sur", "par", "une", "qui", "que",
    "est", "son", "ses", "moi", "lui", "leur", "tout", "cette", "aussi",
    "donne", "mois", "references", "documents", "trouve", "trouver",
    "parle", "concernant", "cela", "ceci", "avoir", "etre", "faire",
    "peux", "mots", "toute", "base", "donnees", "infos", "informations",
    "passages", "concernant",
    # FR — mots vagues fréquents dans les questions de suivi
    "plus", "detail", "details", "meme", "encore", "bien", "tres",
    "non", "oui", "comment", "quoi", "quel", "quelle", "quels", "quelles",
    "dire", "dis", "donner", "expliquer", "voir", "connais", "connaitre",
    "peut", "faut", "donc", "alors", "mais", "car", "donc", "ainsi",
    # EN
    "the", "and", "for", "with", "that", "this", "from", "about",
    "more", "detail", "tell", "give", "show", "explain", "what", "how",
}


def _keyword_fallback_search(db, question: str, k: int) -> list:
    """
    Recherche exacte via ChromaDB where_document lorsque la recherche sémantique
    ne trouve rien de pertinent.
    Désactivé si VECTOR_DB=qdrant (where_document non disponible avec Qdrant).

    Extrait les mots significatifs de la question et cherche les chunks qui les
    contiennent littéralement (plusieurs variantes de casse tentées).
    Exemple : question="l'AMDEC" → cherche "AMDEC", "amdec", "Amdec" dans les chunks.
    """
    from langchain_core.documents import Document

    if VECTOR_DB == "qdrant" or not hasattr(db, "_collection"):
        return []  # where_document non disponible avec Qdrant

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


def _substituer_parent_chunks(resultats: list) -> list:
    """
    B3 — Remplace le page_content de chaque child chunk par son parent_text.
    Le LLM reçoit le parent (800-1200 tokens) au lieu du child (250 tokens).
    Désactivé si USE_PARENT_CHILD=false (pas de parent_text en metadata).

    Déduplication par parent_id : si plusieurs children pointent vers le même parent,
    on ne garde qu'une seule occurrence (meilleur score) pour éviter de répéter
    le même texte dans le contexte LLM.
    """
    if not USE_PARENT_CHILD:
        return resultats
    from langchain_core.documents import Document

    # Passe 1 — substitution + collecte par parent_id
    sans_parent = []        # chunks sans parent_text (ordre conservé)
    parents_vus: dict[str, tuple] = {}  # parent_id → (doc, score) meilleur score

    for doc, score in resultats:
        parent_text = doc.metadata.get("parent_text")
        parent_id = doc.metadata.get("parent_id")
        if parent_text and parent_id:
            if parent_id not in parents_vus or score > parents_vus[parent_id][1]:
                parents_vus[parent_id] = (
                    Document(page_content=parent_text, metadata=doc.metadata),
                    score,
                )
        else:
            sans_parent.append((doc, score))

    # Passe 2 — reconstituer dans l'ordre score décroissant
    parents_dedup = sorted(parents_vus.values(), key=lambda x: x[1], reverse=True)
    enrichis = sorted(sans_parent + parents_dedup, key=lambda x: x[1], reverse=True)

    doublons = len(resultats) - len(sans_parent) - len(parents_vus)
    logger.info(
        f"Parent/child : {len(parents_vus)} parents uniques "
        f"({doublons} doublon(s) supprimé(s))"
    )
    return enrichis


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

    USE_RERANKER_GPU=true : charge le reranker sur GPU en fp16 (~570MB VRAM).
    Vérifier que qwen3.5:4b (5.17GB) + reranker fp16 (0.57GB) = 5.74GB < 6GB VRAM dispo.
    Fallback CPU automatique si CUDA indisponible.
    """
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance
    if not USE_RERANKER:
        return None
    try:
        from FlagEmbedding import FlagReranker
        if USE_RERANKER_GPU:
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
            if use_gpu:
                _reranker_instance = FlagReranker(RERANKER_MODEL, use_fp16=True)
                logger.info(f"Reranker initialisé : {RERANKER_MODEL} (GPU, fp16)")
            else:
                logger.warning("USE_RERANKER_GPU=true mais CUDA indisponible — fallback CPU")
                _reranker_instance = FlagReranker(RERANKER_MODEL, use_fp16=False, devices="cpu")
                logger.info(f"Reranker initialisé : {RERANKER_MODEL} (CPU, fp32)")
        else:
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
        # A1 — log détaillé par chunk
        for rank, (doc, score) in enumerate(reranked, 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            logger.debug(f"  ColBERT #{rank} score={score:.4f} | {source} p.{page}")
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
        # A1 — log détaillé par chunk
        for rank, (doc, score) in enumerate(top, 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            logger.debug(f"  Reranker #{rank} score={score:.4f} | {source} p.{page}")
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
    if not history:
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
                 collection_manager: CollectionManager | None = None,
                 hyde_mode: str = "narrative"):
        self.cm = collection_manager or CollectionManager()
        self.nom_collection = nom_collection
        self.hyde_mode = hyde_mode  # "narrative" (défaut) ou "composition" (RFQPlanner)
        # Auto-select prompt VLM si collection VLM et prompt non spécifié explicitement
        if prompt_name == "defaut" and "vlm" in nom_collection.lower():
            prompt_name = "vlm_robotics"
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
          num_ctx = K × tokens_par_chunk + 1200 overhead, borné entre 4096 et HARD_MAX

        Tokens par chunk :
          USE_PARENT_CHILD=false : ~250 tokens (chunk standard 450 tokens, ~250 utiles)
          USE_PARENT_CHILD=true  : ~1000 tokens (parent chunk envoyé au LLM)

        Exemples :
          48  chunks (child) → K=6,  num_ctx=4096
          446 chunks (child) → K=20, num_ctx=6200 (→ 8192 par l'arrondi)
          446 chunks (parent)→ K=20, num_ctx=21200 (→ 32768 sur RTX 5090, capé 8192 sur 3060)
        """
        try:
            # Qdrant : QdrantCollectionStore.count(), ChromaDB : db._collection.count()
            if VECTOR_DB == "qdrant" and hasattr(self.db, "count"):
                nb_chunks = self.db.count()
            elif hasattr(self.db, "_collection"):
                nb_chunks = self.db._collection.count()
            else:
                nb_chunks = 0
        except Exception:
            nb_chunks = 0

        if nb_chunks == 0:
            return NB_CHUNKS_RECHERCHE, NUM_CTX_MIN

        k = max(6, min(nb_chunks // 8, 20))
        if USE_PARENT_CHILD:
            tokens_par_chunk = 1000  # parent chunks ~1000 tokens (contexte LLM large)
        else:
            tokens_par_chunk = CHUNK_SIZE_APPROX // 4   # ~250 tokens
        overhead = 1200  # prompt + historique + question
        num_ctx_estime = k * tokens_par_chunk + overhead
        num_ctx = _bracket_num_ctx(num_ctx_estime)
        return k, num_ctx

    def rechercher(self, question: str, k: int = NB_CHUNKS_RECHERCHE,
                   history: str = "") -> tuple[str, list[dict]]:
        """
        Recherche les chunks les plus pertinents.

        Pipeline :
          1. Détecte si la question cible une machine spécifique → filtre metadata
          2. Calcule k_candidats (réduit par query si multi-query)
          3a. A6 HyDE — doc hypothétique comme query vectorielle (si USE_HYDE=true)
          3b. A5 Multi-query — N variantes → N recherches vectorielles → RRF fusion
              Sinon : 1 recherche vectorielle classique
          4. Hybrid BM25 — fusion avec RRF sur la query originale
          5. Reranking via cross-encoder BGE (ou ColBERT si USE_COLBERT=true)
          5a. A7 CRAG light — évalue qualité top 2, re-query keyword si score < seuil
          5b. Keyword fallback — si aucun résultat pertinent
          5c. A4 MMR — diversification après reranking (si USE_MMR=true)
          6. Seuil relatif : écarte les chunks au score < score_max × 0.1
          6b. Seuil absolu ≥ 0.01
          7. Déduplication PDF/DOCX
          7a. A2 Cap par source (si MAX_CHUNKS_PER_SOURCE > 0)
          7b. Tail extension
          7c. Section complète
          A1. Log des chunks retenus finaux (niveau DEBUG)
          8. Retourne (contexte_texte, liste_sources)
        """
        # 1. Pré-filtrage metadata
        filtre = _extraire_filtre_question(question)
        if filtre:
            logger.info(f"Filtre metadata : {filtre}")

        # 2. Nombre de candidats à récupérer
        reranker = _get_reranker()
        # A5 : si multi-query actif, réduire k par query (k×2 au lieu de k×3)
        # pour garder ~24 candidats totaux sur 3 queries
        if USE_MULTI_QUERY:
            k_par_query = min(k * 2, RERANKER_CANDIDATS_MAX)
        else:
            k_par_query = min(k * RERANKER_CANDIDATS_MULT, RERANKER_CANDIDATS_MAX) if reranker else k
        k_candidats = min(k * RERANKER_CANDIDATS_MULT, RERANKER_CANDIDATS_MAX) if reranker else k

        # 3a. A6 HyDE — générer un document hypothétique comme query vectorielle principale
        if USE_HYDE:
            query_vecteur = _generer_document_hypothetique(
                question, history=history, filtre=filtre,
                nom_collection=self.nom_collection, hyde_mode=self.hyde_mode,
            )
        else:
            query_vecteur = question

        # 3b. A5 Multi-query — N variantes de queries → N recherches → RRF fusion
        if USE_MULTI_QUERY:
            # Si HyDE actif, passer le doc hypothétique pour inspirer les variantes
            hyde_doc = query_vecteur if USE_HYDE and query_vecteur != question else ""
            variantes = _generer_variantes_question(question, history, hyde_doc=hyde_doc)
            # Si HyDE actif en mode narrative, l'ajouter comme variante supplémentaire.
            # En mode composition (RFQPlanner), NE PAS ajouter : le doc HyDE est une BOM
            # dimension-spécifique → l'ajouter comme variante vectorielle polluerait les résultats
            # avec des termes hors-scope d'autres dimensions.
            if USE_HYDE and query_vecteur != question and self.hyde_mode != "composition":
                variantes = variantes + [query_vecteur]

            result_lists = []
            for q_variante in variantes:
                try:
                    r = self.db.similarity_search_with_score(q_variante, k=k_par_query, filter=filtre)
                    if not r and filtre:
                        r = self.db.similarity_search_with_score(q_variante, k=k_par_query)
                    result_lists.append(r)
                except Exception as e:
                    logger.warning(f"Multi-query recherche '{q_variante[:40]}…' échouée ({e})")
                    try:
                        result_lists.append(
                            self.db.similarity_search_with_score(q_variante, k=k_par_query)
                        )
                    except Exception:
                        result_lists.append([])

            resultats = _rrf_fusion_lists(result_lists, k_final=k_candidats)
        else:
            # 3. Recherche vectorielle classique (query_vecteur = HyDE ou originale)
            try:
                resultats = self.db.similarity_search_with_score(query_vecteur, k=k_candidats, filter=filtre)
                if not resultats and filtre:
                    logger.info(f"Aucun résultat avec filtre {filtre}, retry sans filtre")
                    resultats = self.db.similarity_search_with_score(query_vecteur, k=k_candidats)
            except Exception as e:
                logger.warning(f"Recherche avec filtre échouée ({e}) — retry sans filtre")
                resultats = self.db.similarity_search_with_score(query_vecteur, k=k_candidats)

        # 4. Hybrid BM25 — fusion avec RRF sur la query originale (pas HyDE/variantes)
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

        # 5a. A7 CRAG light — évalue qualité des top 2 chunks, re-query si insuffisant
        if USE_CRAG and resultats:
            top2 = resultats[:2]
            scores_crag = []
            for doc, _ in top2:
                try:
                    prompt_crag = (
                        f"Question : {question}\n\n"
                        f"Passage : {doc.page_content[:500]}\n\n"
                        "Ce passage répond-il à la question ? "
                        "Score de 0 à 1 (0=pas du tout, 1=parfaitement). "
                        "Réponds UNIQUEMENT avec le score numérique."
                    )
                    if ENABLE_NO_THINK:
                        prompt_crag = "/no_think\n\n" + prompt_crag
                    payload_crag = {
                        "model": OLLAMA_MODEL,
                        "prompt": prompt_crag,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_ctx": NUM_CTX_MIN},
                    }
                    resp = requests.post(OLLAMA_API_GENERATE, json=payload_crag, timeout=30)
                    text = resp.json().get("response", "").strip()
                    m = re.search(r'\d+\.?\d*', text)
                    if m:
                        scores_crag.append(min(1.0, float(m.group())))
                except Exception as e:
                    logger.debug(f"CRAG éval chunk échouée : {e}")
            if scores_crag:
                avg_crag = sum(scores_crag) / len(scores_crag)
                logger.info(f"CRAG : score moyen = {avg_crag:.2f} (seuil={CRAG_QUALITY_THRESHOLD})")
                if avg_crag < CRAG_QUALITY_THRESHOLD:
                    logger.info("CRAG : qualité insuffisante → re-query keyword fallback déclenché")
                    fallback_crag = _keyword_fallback_search(self.db, question, k=k)
                    if fallback_crag:
                        resultats = fallback_crag + [r for r in resultats if r not in fallback_crag]

        # 5b. Keyword fallback — si le reranker (ou la RRF sans reranker) ne trouve
        # rien de pertinent, recherche exacte par mots-clés via where_document.
        score_best = resultats[0][1] if resultats else 0.0
        if score_best < KEYWORD_FALLBACK_THRESHOLD:
            fallback = _keyword_fallback_search(self.db, question, k=k)
            if fallback:
                # On prepend les résultats keyword : ils ont une correspondance exacte
                resultats = fallback + [r for r in resultats if r not in fallback]

        # 5c. A4 MMR — diversification après reranking, avant seuil relatif
        if USE_MMR and resultats:
            resultats = _appliquer_mmr(resultats, lambda_mult=MMR_LAMBDA, k=k)

        # 6. Seuil reranker adaptatif : élimine les chunks trop éloignés du meilleur.
        # Multiplicateur adapté au score top :
        #   score_top ≥ 0.8 (collection très spécialisée) → seuil × 0.35
        #   score_top ≥ 0.5                               → seuil × 0.20
        #   score_top < 0.5 (retrieval difficile)         → seuil × 0.10
        # Empêche de garder 17 chunks quand tout est pertinent (→ context truncation).
        if reranker and resultats:
            score_max_r = resultats[0][1]
            if score_max_r >= 0.8:
                mult = 0.35
            elif score_max_r >= 0.5:
                mult = 0.20
            else:
                mult = 0.10
            seuil_relatif = score_max_r * mult
            nb_avant = len(resultats)
            resultats = [(doc, s) for doc, s in resultats if s >= seuil_relatif]
            nb_filtres = nb_avant - len(resultats)
            if nb_filtres:
                logger.info(
                    f"Seuil relatif reranker (×{mult}, ≥{seuil_relatif:.3f}) : "
                    f"{nb_filtres} chunk(s) écarté(s) sur {nb_avant}"
                )

        # 6b. Seuil absolu adaptatif : élimine les chunks quasi-nuls.
        # Quand le top score est très bas (query difficile / cold retrieval), on abaisse
        # le seuil proportionnellement pour ne pas éliminer des chunks légitimement pertinents.
        # Exemples : top=0.8 → seuil=0.01 | top=0.05 → seuil=0.005 | top=0.014 → seuil=0.003
        if resultats:
            score_top_absolu = max(s for _, s in resultats)
            seuil_absolu = min(0.01, score_top_absolu * 0.3)
        else:
            seuil_absolu = 0.01
        nb_avant_absolu = len(resultats)
        resultats = [(doc, s) for doc, s in resultats if s >= seuil_absolu]
        if len(resultats) < nb_avant_absolu:
            logger.info(
                f"Seuil absolu (≥{seuil_absolu:.4f}) : {nb_avant_absolu - len(resultats)} "
                f"chunk(s) quasi-nuls éliminés"
            )

        # 7. Déduplication PDF/DOCX
        resultats = _deduplicater_pdf_docx(resultats)

        # 7a. A2 Cap par source (après reranking et déduplication)
        resultats = _cap_par_source(resultats, MAX_CHUNKS_PER_SOURCE)

        # 7a2. Filtre chunks image-seule (titre + [IMAGE] sans texte extracté)
        # TODO: désactiver quand OCR intégré (Docling VLM / Tesseract) — voir _filtrer_chunks_image_seule
        resultats = _filtrer_chunks_image_seule(resultats)

        # 7b. B3 Parent/child — remplace le child par son parent_text pour le contexte LLM
        # (tail extension et section retrieval ci-dessous sont skipées si USE_PARENT_CHILD=true)
        resultats = _substituer_parent_chunks(resultats)

        # 7c. Tail extension — complète les chunks qui s'arrêtent au milieu d'une liste
        # Désactivée si USE_PARENT_CHILD=true ou VECTOR_DB=qdrant
        resultats = _ajouter_tail_suivant(self.db, resultats)

        # 7d. Section complète — détection intelligente via analyse des résultats initiaux
        # Désactivée si USE_PARENT_CHILD=true ou VECTOR_DB=qdrant
        titre_cible = _detecter_section_ciblee(self.db, question, resultats)
        if titre_cible:
            logger.info(f"Section ciblée détectée : '{titre_cible}' → recherche section complète")
            section_complete = _recuperer_section_complete(self.db, titre_cible, k_max=k*2)
            if section_complete and len(section_complete) > len(resultats):
                nb_avant = len(resultats)
                resultats = section_complete
                logger.info(f"Section complète récupérée : {len(section_complete)} chunks (remplace {nb_avant} résultats)")

        # Hard cap final : k chunks max pour éviter le context truncation
        # _adapter_parametres calcule num_ctx = k × 250 tokens + 1200 overhead.
        # Si les chunks sont plus longs (~350 tokens), on dépasse le KvSize d'Ollama.
        # Le cap garantit que les logs "truncating input prompt" disparaissent.
        if len(resultats) > k:
            resultats = resultats[:k]

        # A1 — Log des chunks retenus finaux avec aperçu contenu (top 3)
        logger.info(f"Pipeline RAG : {len(resultats)} chunk(s) retenus pour le contexte LLM")
        for rank, (doc, score) in enumerate(resultats, 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            machine = doc.metadata.get("machine", "")
            section = doc.metadata.get("hierarchy_parents", "")
            apercu = doc.page_content[:150].replace("\n", " ").strip()
            logger.info(
                f"  Chunk #{rank} score={score:.4f} | {source} p.{page}"
                + (f" | machine={machine}" if machine else "")
                + (f" | section={section}" if section else "")
            )
            if rank <= 3:
                logger.info(f"    ↳ \"{apercu}…\"")

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

            parent_text = meta.get("parent_text")
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
                "parent_text": parent_text,
            })

        return {
            "original_question": question,
            "rewritten_query": rewritten_query,
            "chunks": chunks,
        }

    def generer_avec_sources(self, question: str, stream: bool = True, history: str = "") -> dict:
        """
        Recherche + génération LLM.

        Retourne {
            "reponse":       generator|str,
            "sources":       list[dict],
            "context":       str,          # contexte complet (avec historique)
            "context_chunks": list[str],   # chunks individuels (pour RAGAS)
            "metrics":       dict,         # Level-A metrics (retrieval_ms, scores, hashes)
        }
        """
        k, num_ctx = self._adapter_parametres()

        # Réécrire la question en query autonome si un historique est disponible
        query_recherche = _reformuler_question(question, history)

        t0 = time.monotonic()
        contexte, sources = self.rechercher(query_recherche, k=k, history=history)
        retrieval_ms = (time.monotonic() - t0) * 1000

        # Découper le contexte en chunks individuels AVANT d'ajouter l'historique
        # (pour RAGAS qui a besoin de chunks séparés comme retrieved_contexts)
        context_chunks = [c for c in contexte.split("\n\n---\n\n") if c.strip()]

        # Métriques Level-A
        scores = [s.get("score", 0) for s in sources if "score" in s]
        metrics = {
            "chunks_used": len(context_chunks),
            "top_score": round(max(scores), 3) if scores else None,
            "min_score": round(min(scores), 3) if scores else None,
            "retrieval_ms": round(retrieval_ms, 1),
            "pipeline_hash": _get_pipeline_hash(),
            "search_hash": build_search_config_hash(),
        }

        # Ajouter l'historique au contexte si fourni
        if history:
            contexte = f"Historique de conversation:\n{history}\n\n---\n\n{contexte}"

        prompt = self.prompt_template.format(context=contexte, question=question)

        # Ajuster num_ctx sur la longueur réelle du prompt.
        # ~4 chars/token + 600 tokens de marge pour la réponse générée.
        # On arrondit au bracket fixe (_NUM_CTX_BRACKETS) pour éviter de déclencher
        # un rechargement du modèle Ollama si le bracket ne change pas par rapport
        # au call précédent (query rewriting utilise le même bracket bas).
        tokens_prompt_estimes = len(prompt) // 4
        num_ctx_reel = _bracket_num_ctx(tokens_prompt_estimes + 600)
        if num_ctx_reel != num_ctx:
            logger.info(
                f"num_ctx ajusté : {num_ctx} → {num_ctx_reel} "
                f"(prompt ~{tokens_prompt_estimes} tokens, bracket {num_ctx_reel})"
            )
        num_ctx = num_ctx_reel

        reponse = self._appeler_ollama(prompt, stream=stream, num_ctx=num_ctx)
        return {
            "reponse": reponse,
            "sources": sources,
            "context": contexte,
            "context_chunks": context_chunks,
            "metrics": metrics,
        }

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
            # A8 — filtre optionnel des blocs <think>...</think>
            # Activé via FILTER_THINK_FROM_STREAM=true.
            # qwen3.5:4b peut streamer les tags caractère par caractère →
            # un buffer accumulateur est nécessaire pour détecter les tags complets.
            OPEN_TAG = "<think>"
            CLOSE_TAG = "</think>"
            # Longueur minimale du buffer tail pour détecter un tag partiel
            OPEN_PREFIX_LEN = len(OPEN_TAG) - 1   # 6 chars = "<think"
            CLOSE_PREFIX_LEN = len(CLOSE_TAG) - 1  # 7 chars = "</think"

            def raw_token_gen():
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
                                return
                except requests.exceptions.ChunkedEncodingError as e:
                    logger.warning(f"Stream interrompu : {e}")
                finally:
                    reponse.close()

            if not FILTER_THINK_FROM_STREAM:
                yield from raw_token_gen()
                return

            buffer = ""
            in_think = False

            for token in raw_token_gen():
                buffer += token

                # Traiter le buffer jusqu'à ce qu'il ne puisse plus progresser
                while buffer:
                    if in_think:
                        close_idx = buffer.find(CLOSE_TAG)
                        if close_idx != -1:
                            # Sortie du bloc think : garder tout ce qui suit </think>
                            buffer = buffer[close_idx + len(CLOSE_TAG):]
                            in_think = False
                            # Reboucler pour traiter le reste du buffer
                        else:
                            # Encore dans think : garder seulement le suffixe
                            # potentiellement partiel (pour détecter </think> au prochain token)
                            if len(buffer) > CLOSE_PREFIX_LEN:
                                buffer = buffer[-CLOSE_PREFIX_LEN:]
                            break
                    else:
                        open_idx = buffer.find(OPEN_TAG)
                        if open_idx != -1:
                            # Émettre tout ce qui précède <think>
                            if open_idx > 0:
                                yield buffer[:open_idx]
                            buffer = buffer[open_idx + len(OPEN_TAG):]
                            in_think = True
                            # Reboucler pour traiter la suite (peut-être </think> immédiat)
                        else:
                            # Pas de <think> complet — émettre la partie sûre
                            # (garder le suffixe qui pourrait être un "<think" partiel)
                            safe_len = len(buffer) - OPEN_PREFIX_LEN
                            if safe_len > 0:
                                yield buffer[:safe_len]
                                buffer = buffer[safe_len:]
                            break

            # Vider le buffer restant si on n'est pas dans un bloc think
            if buffer and not in_think:
                yield buffer

        return _stream_tokens()
