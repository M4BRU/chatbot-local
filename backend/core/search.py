"""
core/search.py — RAGEngine : recherche similarité + génération Ollama streaming.
"""

import json
from pathlib import Path

import requests

from core.collection_manager import CollectionManager
from core.embeddings import OLLAMA_API_GENERATE, OLLAMA_MODEL

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
- Ne jamais inventer de spécifications techniques."""

# Prompts nommés disponibles
PROMPTS = {
    "defaut": PROMPT_DEFAUT,
    "vlm_robotics": PROMPT_VLM_ROBOTICS,
    "vlm": PROMPT_VLM_ROBOTICS,  # alias : collection "vlm" → prompt VLM Robotics
}

NB_CHUNKS_RECHERCHE = 6   # fallback si collection vide
CHUNK_SIZE_APPROX = 1000  # doit correspondre à document_manager.CHUNK_SIZE
NUM_CTX_MIN = 4096
NUM_CTX_MAX = 32768


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

    def rechercher(self, question: str, k: int = NB_CHUNKS_RECHERCHE) -> tuple[str, list[dict]]:
        """
        Recherche les chunks les plus pertinents.
        Retourne (contexte_texte, liste_sources).
        """
        resultats = self.db.similarity_search_with_score(question, k=k)

        contexte_parts = []
        sources = []
        sources_vues = set()

        for doc, score in resultats:
            contexte_parts.append(doc.page_content)
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

    def generer_avec_sources(self, question: str, stream: bool = True, history: str = "") -> dict:
        """
        Recherche + génération LLM.

        Retourne {"reponse": generator|str, "sources": list[dict]}
        """
        k, num_ctx = self._adapter_parametres()
        contexte, sources = self.rechercher(question, k=k)

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
                timeout=300,
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
            for ligne in reponse.iter_lines():
                if ligne:
                    donnees = json.loads(ligne)
                    token = donnees.get("response", "")
                    if token:
                        yield token
                    if donnees.get("done", False):
                        break

        return _stream_tokens()
