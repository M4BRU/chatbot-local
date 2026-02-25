"""
core/document_manager.py — Indexation incrémentale des documents.

Tracking via metadata.json par collection (hash SHA256, date, chunk_ids).
Les fichiers metadata sont stockés dans METADATA_DIR (variable d'env).
"""

import hashlib
import json
import logging
import os
import re
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.collection_manager import CollectionManager
from core.parsers import parser_document

logger = logging.getLogger(__name__)

# Taille en tokens (pas en caractères) — mxbai-embed-large : 512 tokens max.
# 450 tokens = sweet spot benchmarks RAG sur docs techniques, avec marge sous la limite.
CHUNK_SIZE_TOKENS = int(os.environ.get("CHUNK_SIZE_TOKENS", "450"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "67"))  # ~15%
MAX_CHUNK_TOKENS = 490  # seuil de protection (< 512 limite modèle)

# Modèle HuggingFace pour le tokenizer (doit correspondre à OLLAMA_EMBED_MODEL)
_EMBED_HF_MODEL = os.environ.get("EMBED_HF_MODEL", "mixedbread-ai/mxbai-embed-large-v1")


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Charge le tokenizer de l'embedding model. Fallback caractères si indisponible."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(_EMBED_HF_MODEL)
        logger.info(f"Tokenizer chargé : {_EMBED_HF_MODEL}")
        return tok
    except Exception as e:
        logger.warning(f"Tokenizer indisponible ({e}) — fallback 1 token ≈ 3 chars")
        return None


def _count_tokens(text: str) -> int:
    """Compte les tokens du texte selon le tokenizer de l'embedding model."""
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return len(text) // 3  # estimation conservative pour texte technique FR

# ── Semantic Chunking ─────────────────────────────────────────────────────────
# Découpe le texte aux frontières sémantiques (changement de sujet) au lieu
# de couper arbitrairement à N caractères.
# Utilise les embeddings nomic-embed-text pour détecter les ruptures sémantiques.
# Plus lent à l'ingest (N appels Ollama par document), mais chunks plus cohérents.
USE_SEMANTIC_CHUNKING = os.environ.get("USE_SEMANTIC_CHUNKING", "false").lower() == "true"
# Seuil percentile : 95 = on coupe seulement aux ruptures très nettes
SEMANTIC_BREAKPOINT_THRESHOLD = int(os.environ.get("SEMANTIC_BREAKPOINT_THRESHOLD", "95"))

# Répertoire de stockage des metadata (séparé de ChromaDB)
METADATA_DIR = Path(os.environ.get("METADATA_DIR", "./documents_metadata"))

# ── Contextual Retrieval ──────────────────────────────────────────────────────
# Enrichit chaque chunk avec son contexte dans le document avant indexation.
# Réduit les erreurs de retrieval de ~67% (technique Anthropic).
# Désactivé par défaut : nécessite N appels Ollama par document à l'ingest.
USE_CONTEXTUAL_RETRIEVAL = os.environ.get("USE_CONTEXTUAL_RETRIEVAL", "false").lower() == "true"
CONTEXTUAL_MAX_WORKERS = int(os.environ.get("CONTEXTUAL_MAX_WORKERS", "4"))
# Service Ollama dédié au contextual retrieval (CPU uniquement — ne touche pas à la VRAM du LLM principal)
CONTEXTUAL_OLLAMA_URL = os.environ.get(
    "CONTEXTUAL_OLLAMA_URL",
    os.environ.get("OLLAMA_URL", "http://localhost:11434")  # fallback sur le principal si pas de service dédié
)
CONTEXTUAL_MODEL = os.environ.get("CONTEXTUAL_MODEL", "qwen3:0.6b")
_CONTEXTUAL_NO_THINK = os.environ.get("ENABLE_NO_THINK", "false").lower() == "true"
_CONTEXTUAL_PROMPT = (
    "Tu es un assistant technique. Voici un document :\n"
    "<document>\n{document}\n</document>\n\n"
    "Voici un extrait de ce document :\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Génère en 1-2 phrases le contexte de cet extrait : quelle machine est concernée, "
    "quel sujet ou quelle section. Réponds uniquement avec ce contexte, sans introduction."
)


def _construire_hierarchie_parents(tous_morceaux_avec_page: list) -> list[list[str]]:
    """
    Construit la hiérarchie des parents pour chaque chunk en analysant les labels Docling.

    Retourne une liste de listes : hierarchies[i] = liste des titres parents du chunk i.
    Exemple : ["CHAPITRE 1", "Article 1.1", "Sous-section A"]
    """
    hierarchies = []
    pile_titres = {}  # niveau -> titre, maintient les titres actuels par niveau

    for page, morceau in tous_morceaux_avec_page:
        parents_actuels = []

        # Analyser ce chunk pour détecter s'il contient un nouveau titre
        lignes = morceau.split('\n')[:5]  # Premières lignes seulement
        nouveau_titre = None
        nouveau_niveau = None

        for ligne in lignes:
            ligne = ligne.strip()
            # Détecter les labels Docling [TITRE-L1], [SECTION-L2], etc.
            match = re.match(r'^\[(TITRE|SECTION)-L(\d+)\]\s*(.+)', ligne, re.IGNORECASE)
            if match:
                label_type, niveau_str, titre_text = match.groups()
                niveau = int(niveau_str)
                titre_clean = titre_text.strip()
                if len(titre_clean) > 3:
                    nouveau_titre = titre_clean
                    nouveau_niveau = niveau
                    break

        # Si on trouve un nouveau titre, mettre à jour la pile
        if nouveau_titre and nouveau_niveau is not None:
            # Supprimer tous les niveaux >= au nouveau niveau (fermer les sections)
            niveaux_a_supprimer = [n for n in pile_titres.keys() if n >= nouveau_niveau]
            for n in niveaux_a_supprimer:
                del pile_titres[n]

            # Ajouter le nouveau titre
            pile_titres[nouveau_niveau] = nouveau_titre

        # Construire la liste des parents actuels (du plus haut niveau au plus bas)
        for niveau in sorted(pile_titres.keys()):
            if niveau < (nouveau_niveau or float('inf')):  # Exclure le titre du chunk actuel
                parents_actuels.append(pile_titres[niveau])

        hierarchies.append(parents_actuels)

    return hierarchies


def _enrichir_chunk_contexte(chunk: str, document_complet: str) -> str:
    """
    Appelle le service Ollama contextuel pour générer un contexte spécifique à ce chunk.
    Retourne le chunk suffixé par son contexte, ou le chunk original en cas d'échec.
    """
    prompt = _CONTEXTUAL_PROMPT.format(document=document_complet[:6000], chunk=chunk)
    try:
        resp = requests.post(
            f"{CONTEXTUAL_OLLAMA_URL}/api/generate",
            json={
                "model": CONTEXTUAL_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,  # paramètre API officiel Ollama — désactive le reasoning proprement
                "options": {"temperature": 0, "num_predict": 200},
            },
            timeout=120,
        )
        resp.raise_for_status()
        contexte = resp.json().get("response", "").strip()
        if contexte and len(contexte) >= 20:
            return f"{chunk}\n\n[{contexte}]"
    except Exception as e:
        logger.warning(f"Contextual retrieval chunk échoué : {e}")
    return chunk

def detect_hierarchy_patterns(text: str) -> list[dict]:
    """
    Détecte les patterns hiérarchiques dans le texte avec validation IA flexible.

    Returns:
        list[dict]: [{"level": str, "title": str, "line_num": int, "validated": bool}, ...]
    """
    import re

    # Patterns génériques
    patterns = [
        (r'^(\d+\.)\s+(.+)', 'numbered_section'),
        (r'^(\d+\.\d+\.)\s*(.*)$', 'numbered_subsection'),
        (r'^(\d+\.\d+\.\d+\.)\s+(.+)', 'numbered_subsubsection'),
        (r'^-(\d+)\s+(.+)', 'numbered_bullet'),
        (r'^\s*-\s+(.+)', 'bullet'),
        (r'^\s*▪\s+(.+)', 'sub_bullet'),
    ]

    hierarchy = []
    lines = text.split('\n')
    total_lines = len(lines)

    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 3:
            continue

        # D'abord vérifier les patterns explicites (numérotés, bullets)
        matched_explicit = False
        for pattern, level in patterns:
            match = re.match(pattern, line_clean)
            if match:
                title = match.group(2).strip() if len(match.groups()) >= 2 else match.group(1).strip()
                hierarchy.append({
                    'level': level,
                    'title': title,
                    'line_num': i,
                    'pattern': match.group(0),
                    'validated': True
                })
                matched_explicit = True
                break

        # Si pas de pattern explicite, vérifier si c'est un titre potentiel
        if not matched_explicit and _is_potential_title(line_clean, i, total_lines):
            context_before = '\n'.join(lines[max(0, i-2):i]) if i > 0 else ""
            context_after = '\n'.join(lines[i+1:min(len(lines), i+3)]) if i < len(lines)-1 else ""

            is_title = _validate_title_with_ai(line_clean, context_before, context_after)
            if is_title:
                hierarchy.append({
                    'level': 'section_title',
                    'title': line_clean,
                    'line_num': i,
                    'pattern': line_clean,
                    'validated': True
                })

    return hierarchy


def _is_potential_title(text: str, line_num: int, total_lines: int) -> bool:
    """Critères larges pour détecter un titre potentiel."""
    clean_text = text.strip()

    return (
        len(clean_text) >= 3 and
        len(clean_text) <= 60 and           # Titres pas trop longs
        not clean_text.endswith('.') and    # Pas une phrase complète
        not clean_text.endswith(',') and    # Pas au milieu d'une phrase
        len(clean_text.split()) <= 8 and    # Max 8 mots (titre concis)
        not clean_text.lower().startswith(('le ', 'la ', 'les ', 'des ', 'un ', 'une '))  # Pas début d'article
    )


def _validate_title_with_ai(text: str, context_before: str, context_after: str) -> bool:
    """Utilise le LLM contextuel pour valider si un texte est un titre de section."""
    prompt = f'"{text}" - TITRE ou TEXTE ?'

    if _CONTEXTUAL_NO_THINK:
        prompt = "/no_think\n\n" + prompt

    try:
        resp = requests.post(
            f"{CONTEXTUAL_OLLAMA_URL}/api/generate",
            json={
                "model": CONTEXTUAL_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 10},
            },
            timeout=30,
        )
        resp.raise_for_status()
        response = resp.json().get("response", "").strip().upper()
        return "TITRE" in response
    except Exception:
        # Si validation IA échoue, on considère que c'est un titre (fail-safe)
        return True


# Noms de machines VLM Robotics à détecter dans les noms de fichiers
_MACHINES = ["GEMINI", "SOLO", "COMPAQT", "HYMANCO"]

# Mots-clés de type de document (en minuscules)
_MOTS_TYPE_DOC = {"devis", "quotation", "quote", "offre", "dossier", "rfi"}


def _normaliser(texte: str) -> str:
    """Supprime les accents et met en majuscules (pour comparer sans accent)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texte.upper())
        if unicodedata.category(c) != "Mn"
    )


def _extraire_metadata_fichier(nom_fichier: str) -> dict:
    """
    Extrait machine, type_doc, ref_projet et client depuis le nom de fichier.

    Patterns reconnus (best-effort, silencieux si non détecté) :
      "0082 - PRISMA - Gémini VLM Robotics - ind1.pdf"  → client=PRISMA,  machine=GEMINI
      "AP0120 - FAN3D - Devis - Ind05.pdf"              → client=FAN3D,   type_doc=devis
      "Irepa_Laser- AP0082 - Offre VLM Robotics.docx"   → client=Irepa Laser
      "DossierTechnique_SOLO_FR_ind3.pdf"               → machine=SOLO,   type_doc=dossier
    """
    stem = Path(nom_fichier).stem
    stem_norm = _normaliser(stem)
    stem_low = stem.lower()

    # ── Machine ────────────────────────────────────────────────────────
    machine = next((m for m in _MACHINES if m in stem_norm), "")

    # ── Type de document ───────────────────────────────────────────────
    type_doc = ""
    if "devis" in stem_low or "quotation" in stem_low or "quote" in stem_low:
        type_doc = "devis"
    elif "offre" in stem_low:
        type_doc = "offre"
    elif "dossier" in stem_low and "technique" in stem_low:
        type_doc = "dossier_technique"
    elif "rfi" in stem_low:
        type_doc = "rfi"

    # ── Référence projet (AP0xxx ou suite de 4-6 chiffres) ────────────
    ref = ""
    m_ref = re.search(r"(AP\d+|\b\d{4,6}\b)", stem)
    if m_ref:
        ref = m_ref.group(0)

    # ── Client ─────────────────────────────────────────────────────────
    client = ""

    # Pattern 1 : "NNNN - CLIENT - ..." ou "APxxxx - CLIENT - ..."
    m1 = re.match(r"^(?:AP)?\d+\s*-+\s*([A-Za-z][A-Za-z0-9]+)\s*-", stem)
    if m1:
        candidate = m1.group(1).strip()
        if len(candidate) >= 3 and candidate.lower() not in _MOTS_TYPE_DOC:
            client = candidate

    # Pattern 2 : "Nom_Composé- ref..." (ex: "Irepa_Laser- AP0082")
    if not client:
        m2 = re.match(r"^([A-Za-z][A-Za-z]*(?:_[A-Za-z]+)+)\s*[-_ ]+(?:AP|\d)", stem)
        if m2:
            client = m2.group(1).replace("_", " ").strip()

    return {
        "machine": machine,
        "type_doc": type_doc,
        "ref_projet": ref,
        "client": client,
    }


class DocumentManager:
    """Gère l'indexation incrémentale des documents dans les collections."""

    def __init__(self, collection_manager: CollectionManager | None = None):
        self.cm = collection_manager or CollectionManager()

        # Splitter de secours : toujours disponible (protection anti-chunks trop gros)
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=_count_tokens,
        )

        if USE_SEMANTIC_CHUNKING:
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from core.embeddings import get_embeddings
                # Structure-aware semantic chunking avec seuils adaptatifs
                # VARIABLE AJUSTABLE : self.base_threshold contrôle la cohésion sémantique
                # - Plus haut (80-95%) = chunks plus gros, plus conservateur
                # - Plus bas (40-60%) = chunks plus petits, plus fragmentés
                # - Actuellement 75% = bon compromis entre cohésion et granularité
                # Dans le chunking adaptatif : 30% près des titres, 75% ailleurs
                self.base_embeddings = get_embeddings()
                self.base_threshold = 75  # Seuil de base structure-aware (était 60% avant)
                self.splitter = self._recursive_splitter  # Fallback toujours disponible
                logger.info(
                    f"Structure-aware semantic chunking activé (seuil base={self.base_threshold}%)"
                )
            except Exception as e:
                logger.warning(f"SemanticChunker indisponible ({e}) — fallback RecursiveCharacterTextSplitter")
                self.splitter = self._recursive_splitter
                self.base_embeddings = None
                self.base_threshold = None
        else:
            self.splitter = self._recursive_splitter
            self.base_embeddings = None
            self.base_threshold = None

    def _metadata_path(self, nom_collection: str) -> Path:
        return METADATA_DIR / f"{nom_collection}.json"

    def _charger_metadata(self, nom_collection: str) -> dict:
        chemin = self._metadata_path(nom_collection)
        if chemin.exists():
            return json.loads(chemin.read_text(encoding="utf-8"))
        return {"documents": {}}

    def _sauvegarder_metadata(self, nom_collection: str, metadata: dict) -> None:
        chemin = self._metadata_path(nom_collection)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _calculer_hash(chemin: Path) -> str:
        h = hashlib.sha256()
        with open(chemin, "rb") as f:
            for bloc in iter(lambda: f.read(8192), b""):
                h.update(bloc)
        return h.hexdigest()

    def document_est_indexe(self, nom_collection: str, chemin: Path) -> bool:
        """Vérifie si un document est déjà indexé (même hash SHA256)."""
        chemin = Path(chemin)
        metadata = self._charger_metadata(nom_collection)
        doc_info = metadata["documents"].get(chemin.name)
        if not doc_info:
            return False
        return doc_info["sha256"] == self._calculer_hash(chemin)

    def ajouter_document(self, nom_collection: str, chemin: Path, force: bool = False) -> dict:
        """
        Indexe un document dans une collection.

        Retourne un dict : {"status": "indexed"|"skipped", "chunks": int, "message": str}
        """
        chemin = Path(chemin)

        if not force and self.document_est_indexe(nom_collection, chemin):
            return {
                "status": "skipped",
                "chunks": 0,
                "message": f"{chemin.name} : déjà indexé (hash identique)",
            }

        # Parser le document
        pages = parser_document(chemin)
        if not pages:
            return {
                "status": "skipped",
                "chunks": 0,
                "message": f"{chemin.name} : aucun texte extrait",
            }

        # Découper en chunks
        textes = []
        metadonnees = []
        chunk_ids = []

        # Filtrer les champs vides (ChromaDB rejette les chaînes vides)
        meta_fichier = {
            k: v for k, v in _extraire_metadata_fichier(chemin.name).items() if v
        }

        # Contextual retrieval : texte complet du document pour le contexte Ollama
        document_complet = ""
        if USE_CONTEXTUAL_RETRIEVAL:
            document_complet = "\n\n".join(p.texte for p in pages)

        # Collecter tous les morceaux de toutes les pages d'abord
        morceaux_par_page: list[tuple[object, list[str]]] = []
        for page in pages:
            morceaux = self.splitter.split_text(page.texte)
            # Protection : re-découpe tout chunk dépassant MAX_CHUNK_TOKENS (< 512 limite mxbai).
            # Appliqué toujours (SemanticChunker ET RecursiveCharacterTextSplitter peuvent
            # produire des chunks trop grands sur du texte technique dense sans séparateur).
            morceaux_proteges = []
            for m in morceaux:
                if _count_tokens(m) > MAX_CHUNK_TOKENS:
                    morceaux_proteges.extend(self._recursive_splitter.split_text(m))
                else:
                    morceaux_proteges.append(m)
            morceaux = morceaux_proteges
            morceaux_par_page.append((page, morceaux))

        # ── chunk_idx + has_continuation ──────────────────────────────────────
        # Calculé sur le texte RAW (avant enrichissement contextuel) :
        # le préfixe contextuel ajouté ensuite masquerait les patterns de liste.
        # has_continuation=True si chunk N et N+1 sont dans la même liste :
        #   N finit par un item de liste  ET  N+1 commence par un item de liste
        #   ET même fichier source (pas de continuation entre deux documents).
        tous_raw = [(page, m) for page, morceaux in morceaux_par_page for m in morceaux]
        _BULLET = re.compile(r'^[\s]*[▪\-•\*–]\s+\S', re.MULTILINE)
        continuation_flags: list[bool] = []
        for i, (page_i, morceau_i) in enumerate(tous_raw):
            if i + 1 < len(tous_raw):
                page_j, morceau_j = tous_raw[i + 1]
                meme_source = (page_i.source == page_j.source)
                ends_bullet = bool(re.search(r'\n[\s]*[▪\-•\*–]\s+\S[^\n]*$', morceau_i))
                starts_bullet = bool(_BULLET.match(morceau_j))
                continuation_flags.append(meme_source and ends_bullet and starts_bullet)
            else:
                continuation_flags.append(False)

        # Enrichissement contextuel en parallèle si activé
        if USE_CONTEXTUAL_RETRIEVAL and document_complet:
            tous_morceaux = [(page, m) for page, morceaux in morceaux_par_page for m in morceaux]
            nb_total = len(tous_morceaux)
            logger.info(
                f"Contextual retrieval : {nb_total} chunks à enrichir "
                f"({CONTEXTUAL_MAX_WORKERS} workers parallèles)…"
            )
            # Dispatch parallèle — même principe qu'un compute shader :
            # chaque worker traite un chunk indépendamment, on collecte dans l'ordre d'origine
            enrichis: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=CONTEXTUAL_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _enrichir_chunk_contexte, morceau, document_complet
                    ): idx
                    for idx, (_, morceau) in enumerate(tous_morceaux)
                }
                for future in as_completed(futures):
                    enrichis[futures[future]] = future.result()

            # Reconstruire morceaux_par_page avec les chunks enrichis dans le bon ordre
            idx = 0
            morceaux_par_page_enrichis: list[tuple[object, list[str]]] = []
            for page, morceaux in morceaux_par_page:
                enrichis_page = [enrichis[idx + i] for i in range(len(morceaux))]
                morceaux_par_page_enrichis.append((page, enrichis_page))
                idx += len(morceaux)
            morceaux_par_page = morceaux_par_page_enrichis
            logger.info(f"Contextual retrieval terminé : {nb_total} chunks enrichis")

        # Construire la hiérarchie des parents pour chaque chunk
        tous_morceaux_avec_page = [(page, morceau) for page, morceaux in morceaux_par_page for morceau in morceaux]
        hierarchies = _construire_hierarchie_parents(tous_morceaux_avec_page)

        chunk_global_idx = 0
        for page, morceaux in morceaux_par_page:
            for morceau in morceaux:
                cid = str(uuid.uuid4())
                chunk_ids.append(cid)
                textes.append(morceau)

                # Récupérer la hiérarchie de ce chunk
                hierarchy_parents = hierarchies[chunk_global_idx]

                metadata_chunk = {
                    "source": page.source,
                    "page":   page.page,
                    "chunk_idx": chunk_global_idx,
                    "has_continuation": continuation_flags[chunk_global_idx],
                    **meta_fichier,
                }

                # Sérialiser hierarchy_parents en JSON string — ChromaDB refuse les listes
                if hierarchy_parents:
                    metadata_chunk["hierarchy_parents"] = json.dumps(hierarchy_parents)

                metadonnees.append(metadata_chunk)
                chunk_global_idx += 1

        # Supprimer les anciens chunks de ce document si re-indexation
        metadata = self._charger_metadata(nom_collection)
        doc_info = metadata["documents"].get(chemin.name)
        if doc_info and doc_info.get("chunk_ids"):
            try:
                db = self.cm.get_collection(nom_collection)
                db.delete(ids=doc_info["chunk_ids"])
            except Exception:
                pass

        # Ajouter les nouveaux chunks
        db = self.cm.creer_collection(nom_collection)
        db.add_texts(texts=textes, metadatas=metadonnees, ids=chunk_ids)

        # Mettre à jour le metadata.json
        metadata["documents"][chemin.name] = {
            "sha256": self._calculer_hash(chemin),
            "date": datetime.now().isoformat(),
            "chunk_ids": chunk_ids,
            "nb_chunks": len(chunk_ids),
            "nb_pages": len(pages),
        }
        self._sauvegarder_metadata(nom_collection, metadata)

        # Détecter si un fallback parser a été utilisé (pas de labels structurels)
        parsers_used = {p.parser for p in pages}
        warnings = []
        if "pymupdf" in parsers_used:
            warnings.append(
                "Docling a échoué pour ce fichier — fallback PyMuPDF utilisé. "
                "Les chunks n'ont pas de labels structurels ([SECTION], [TEXT], etc.). "
                "Qualité de retrieval potentiellement réduite."
            )

        return {
            "status": "indexed",
            "chunks": len(chunk_ids),
            "message": f"{chemin.name} : {len(chunk_ids)} chunks indexés ({len(pages)} pages)",
            "warnings": warnings,
        }

    def supprimer_document(self, nom_collection: str, nom_fichier: str) -> bool:
        """Supprime un document de la collection (chunks + metadata)."""
        metadata = self._charger_metadata(nom_collection)
        doc_info = metadata["documents"].get(nom_fichier)
        if not doc_info:
            return False

        # Supprimer les chunks de ChromaDB
        if doc_info.get("chunk_ids"):
            try:
                db = self.cm.get_collection(nom_collection)
                db.delete(ids=doc_info["chunk_ids"])
            except Exception:
                pass

        # Retirer du metadata
        del metadata["documents"][nom_fichier]
        self._sauvegarder_metadata(nom_collection, metadata)
        return True

    def lister_documents(self, nom_collection: str) -> list[dict]:
        """Liste les documents indexés dans une collection."""
        metadata = self._charger_metadata(nom_collection)
        docs = []
        for nom, info in metadata["documents"].items():
            docs.append({
                "nom": nom,
                "date": info.get("date", ""),
                "nb_chunks": info.get("nb_chunks", 0),
                "nb_pages": info.get("nb_pages", 0),
                "sha256": info.get("sha256", ""),
            })
        return docs
