"""
core/search.py — RAGEngine : recherche similarité + génération Ollama streaming.
"""

import json
from pathlib import Path

import requests

from core.embeddings import OLLAMA_MODEL, OLLAMA_API_GENERATE
from core.collection_manager import CollectionManager

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
}

NB_CHUNKS_RECHERCHE = 4


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

    def generer_avec_sources(self, question: str, stream: bool = True) -> dict:
        """
        Recherche + génération LLM.

        Retourne {"reponse": generator|str, "sources": list[dict]}
        """
        contexte, sources = self.rechercher(question)
        prompt = self.prompt_template.format(context=contexte, question=question)

        reponse = self._appeler_ollama(prompt, stream=stream)
        return {"reponse": reponse, "sources": sources}

    @staticmethod
    def _appeler_ollama(prompt: str, stream: bool = True):
        """
        Appelle l'API Ollama.
        Si stream=True, retourne un générateur de tokens.
        Si stream=False, retourne la réponse complète (str).
        """
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.3,
                "num_ctx": 4096,
            },
        }

        try:
            reponse = requests.post(
                OLLAMA_API_GENERATE,
                json=payload,
                stream=stream,
                timeout=120,
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
