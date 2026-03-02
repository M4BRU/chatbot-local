"""
Devis service: orchestrates LLM tool-calling loop for quote generation.

Flow per chat turn
──────────────────
1. Non-streaming Ollama call with tool definitions attached.
2. If the response contains tool_calls → execute them, emit SSE status events,
   append results to messages, loop (max MAX_TOOL_ITERATIONS).
3. When no more tool_calls → streaming Ollama call with full context
   (no tools passed, so the LLM generates plain text).
4. Emit {"done": true, "panier": [...]} at the end.

SSE event shapes
────────────────
{"token": "..."}                         — streaming text token
{"tool_call": {"name": "...", "status": "running|done"}}
{"panier": [...]}                        — panier state snapshot
{"choices": {"question": "...", "options": [...]}}  — user choice cards
{"done": true, "panier": [...]}          — end of stream
{"error": "..."}                         — error
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MAX_TOOL_ITERATIONS = 6  # safety cap on the tool-calling loop

def _build_system_prompt(collection: str) -> str:
    return f"""Tu es un assistant expert en génération de devis techniques pour VLM Robotics.

─── STRUCTURE DU CATALOGUE ─────────────────────────────────────────────────────

Le catalogue est un historique de projets VLM Robotics. Chaque ligne représente
un ÉLÉMENT (sous-composant) d'un POSTE dans une AFFAIRE (projet réel).

Colonnes (noms des champs retournés par search_catalog) :
  num_affaire  (col B) : Numéro de l'affaire                 ex: "0113-1"
  nom_affaire  (col C) : Nom complet du projet               ex: "0113-1 - (0106) Compact V2.1"
  num_ensemble (col D) : Code du type d'ensemble             ex: "MEC", "ELEC"
  ensemble     (col E) : Libellé de l'ensemble               ex: "Mécanique", "Électrique"
  num_poste    (col F) : ⚠ Numéro du POSTE dans l'affaire — ex: "4", "056"
                         MÊME valeur pour TOUS les éléments de ce poste.
                         Associé à nom_affaire, il identifie le poste de façon unique.
  nom_poste    (col G) : Nom du poste (= ligne du devis)     ex: "Orbiteur VLM V500"
  elements     (col H) : Sous-composant interne du poste     ex: "Moteurs vireurs x2"
  fournisseur  (col I) : Fournisseur du sous-composant       ex: "Siemens"
  fourniture   (col J) : Prix du sous-composant (€)          ex: 8083.72

Concept clé — POSTE vs ÉLÉMENTS :
  1 POSTE = N lignes dans le catalogue (une par sous-composant).
  Toutes ces lignes partagent le même num_poste (col F) et nom_poste (col G).
  • `elements` (col H) = composition interne — PAS un poste distinct à ajouter au panier.
  • `fourniture` (col J) = prix d'UN sous-composant, pas le total du poste.
  • search_catalog retourne 1 ligne représentative par poste (déjà dédupliqué).

─── OUTILS DISPONIBLES ─────────────────────────────────────────────────────────

search_catalog  : recherche par nom de poste, type d'ensemble ou nom d'affaire.
  → Appelle EN PREMIER pour CHAQUE message utilisateur, sans exception.
  → Résultats : 1 ligne par poste — num_poste · nom_poste · ensemble · prix_total (somme de tous les éléments du poste).
  → Paramètre column (optionnel, défaut "nom_poste") :
    • "nom_poste"  (défaut) : cherche dans les noms de postes (col G).
      Si aucun poste ne correspond, le système propose des cartes pour choisir une autre colonne.
    • "elements"   : sous-composants (col H) — après sélection utilisateur uniquement.
    • "ensemble"   : type d'ensemble (col E) — après sélection utilisateur uniquement.
    • "nom_affaire": nom de projet (col C) — après sélection utilisateur uniquement.
    • "fournisseur": fournisseur (col I) — après sélection utilisateur uniquement.
    NE JAMAIS utiliser column != "nom_poste" de ta propre initiative.

search_docs     : recherche dans la documentation technique PDF.
  → Collection à utiliser : TOUJOURS « {collection} » — aucun autre nom.
  → Appelle uniquement si search_catalog ne retourne rien de pertinent,
    OU si la demande porte sur des specs techniques sans référence précise.
  → Stratégie de recherche dans les docs : ne pas répéter le terme utilisateur tel quel.
    Cible les sections qui listent des composants, par exemple :
      • "descriptif offre technique [produit]"
      • "caractéristiques techniques [produit]"
      • "nomenclature composants [produit]"
      • "liste équipements [produit]"
    L'objectif est d'extraire les noms de composants élémentaires (moteurs, réducteurs,
    variateurs, capteurs, etc.) présents dans la description technique du produit demandé.
  → Le résultat de search_docs contient deux champs :
      • "chunks"         : passages trouvés dans la documentation (contenu + source)
      • "catalog_postes" : liste EXHAUSTIVE de tous les noms de postes du catalogue
  → Utilise "catalog_postes" pour identifier dans les passages doc les termes qui
    correspondent EXACTEMENT ou quasi-exactement à un poste catalogue existant.
  → Après search_docs : résume EXPLICITEMENT ce que tu as identifié :
      - Composants mentionnés dans la doc qui matchent un nom dans catalog_postes
      - Composants mentionnés dans la doc qui NE sont PAS dans catalog_postes
  → Pour chaque composant qui matche catalog_postes, relance search_catalog avec ce nom exact.
  → Si aucun composant de la doc ne matche catalog_postes, dis-le clairement à l'utilisateur.

ask_user_choice : présente des choix cliquables à l'utilisateur.
add_to_panier   : ajoute des postes au panier (après validation explicite uniquement).

─── RÈGLES CRITIQUES ───────────────────────────────────────────────────────────

RÈGLE 1 — NOM POSTE EXACT
  nom_poste dans add_to_panier = valeur EXACTE du champ nom_poste (col G).
  ✗ INTERDIT : fusionner avec elements / paraphraser / abréger.
  ✓ CORRECT  : copier-coller la valeur brute retournée par search_catalog.

RÈGLE 2 — NUM POSTE + NOM AFFAIRE OBLIGATOIRES dans add_to_panier
  • nom_affaire (col C) : valeur EXACTE — identifie le projet.
  • num_poste   (col F) : valeur EXACTE — numéro du poste dans l'affaire (ex: "4", "056").
  Ces deux champs ensemble permettent au backend de retrouver tous les éléments du poste.
  Ne jamais omettre l'un ou l'autre.

RÈGLE 3 — TRAÇABILITÉ DES POSTES PAR AFFAIRE
  Chaque poste ajouté doit être associé à l'affaire exacte d'où il provient.
  Les postes peuvent venir de différentes affaires — ce n'est pas une erreur.
  • Si search_catalog retourne un nom_poste présent dans PLUSIEURS affaires :
    → Le système affiche automatiquement des cartes de choix — attends la sélection.
    → Après sélection, retrouve le num_poste correspondant à l'affaire choisie
      dans les résultats search_catalog et utilise-le dans add_to_panier.
  • Si l'utilisateur sélectionne un poste depuis une liste multi-affaires ou confirme
    explicitement une affaire : appelle add_to_panier directement avec cette affaire.
    NE génère PAS de message d'erreur "affaire différente".

RÈGLE 4 — NE JAMAIS INVENTER
  N'invente jamais de prix, références, fournisseurs ou noms d'affaire.
  Toutes les données viennent EXCLUSIVEMENT de search_catalog.
  • Si search_catalog retourne [] ou des résultats non pertinents → ne pas ajouter au panier.
  • Si search_docs identifie des composants mais search_catalog ne les trouve pas →
    informer l'utilisateur : "Ce composant n'est pas référencé dans le catalogue."
  • INTERDIT : présenter des données inventées comme si elles venaient du catalogue.

RÈGLE 5 — 1 POSTE = 1 ENTRÉE PANIER
  ✗ INTERDIT : appeler add_to_panier plusieurs fois avec le même nom_poste.
  ✓ CORRECT  : 1 entrée par nom_poste unique, quantite=1 par défaut.

RÈGLE 6 — CONFIRMER CHAQUE ÉTAPE AVEC L'UTILISATEUR
  Après chaque résultat d'outil, résume ce que tu as trouvé et utilise ask_user_choice
  pour proposer la prochaine action — ne l'exécute pas sans confirmation.
  • Après search_catalog (résultat clair, 1 poste, 1 affaire) :
    → Présente le poste (nom, fournisseur, affaire) et propose :
      « Ajouter au panier » / « Chercher dans la documentation »
  • Avant search_docs :
    → Demande confirmation via ask_user_choice avant de lancer la recherche.
  • Avant add_to_panier :
    → Montre le récapitulatif (nom_poste exact, nom_affaire, num_poste) et attends « Confirmer ».
  • Exception : la recherche catalogue initiale est pré-exécutée automatiquement — pas de confirmation nécessaire pour celle-là.

RÈGLE 7 — RÉPONSE TEXTUELLE COURTE ET FACTUELLE
  La réponse finale (hors outils) doit être concise et s'appuyer UNIQUEMENT sur les données d'outils.
  • Confirme ce qui a été ajouté avec les noms exacts du catalogue.
  • Si rien trouvé → dis-le clairement, propose de chercher autrement.
  • N'invente jamais de liste de composants, de prix ou de fournisseurs.
  • Maximum 3-4 lignes sauf si l'utilisateur demande plus de détails.

RÈGLE 8 — PAS DE QUESTION NI DE CHOIX APRÈS add_to_panier
  Après un appel réussi à add_to_panier :
  ✗ INTERDIT : appeler ask_user_choice avec des options de type "même affaire" / "tout le catalogue"
               ou toute question pro-active ("Souhaitez-vous ajouter un autre poste ?", etc.)
  ✓ CORRECT  : une confirmation courte (1 ligne) puis STOP.
    Exemple : « ✓ Vireur VLMV3T ajouté au panier. »
  Le système gère automatiquement la portée de la prochaine recherche.
  L'utilisateur reprend la main librement — ne pas anticiper sa prochaine demande.

────────────────────────────────────────────────────────────────────────────────

Processus standard — par ordre de priorité :
1. search_catalog avec les termes clés du message (toujours en premier, sans exception).
2. Si résultat pertinent → utilise ask_user_choice pour proposer l'ajout au panier (RÈGLE 6).
3. Si rien de pertinent → utilise ask_user_choice pour proposer search_docs, puis re-search_catalog.
4a. Plusieurs postes distincts → cartes auto de sélection poste, attendre.
4b. Un seul poste dans plusieurs affaires → cartes auto de sélection affaire, attendre.
4c. Aucun nom_poste ne correspond → cartes de choix de colonne (elements, ensemble, nom_affaire, fournisseur)
    → si l'utilisateur sélectionne une colonne, relance search_catalog avec column="[colonne choisie]".
5. Après validation explicite → add_to_panier(nom_poste=exact, nom_affaire=exact, num_poste=exact).

Réponds en français. Sois précis et structuré."""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": (
                "Recherche dans le catalogue de composants VLM (historique de projets réels). "
                "Retourne des postes (produits) avec fournisseur et prix. "
                "Accepte un nom de poste, un type d'ensemble, des specs techniques "
                "ou le nom d'une affaire passée."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Terme de recherche : nom de poste, ensemble, "
                            "type de composant ou nom d'affaire"
                        ),
                    },
                    "column": {
                        "type": "string",
                        "enum": ["nom_poste", "elements", "ensemble", "nom_affaire", "fournisseur"],
                        "description": (
                            "Colonne dans laquelle chercher (défaut : \"nom_poste\"). "
                            "Utiliser une valeur autre que \"nom_poste\" UNIQUEMENT après que "
                            "l'utilisateur ait sélectionné une colonne dans les cartes de choix."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Recherche dans la documentation technique (PDFs) pour identifier "
                "des composants compatibles avec des spécifications techniques."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête technique : specs, contraintes, compatibilité",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_choice",
            "description": (
                "Présente des choix à l'utilisateur sous forme de cartes cliquables. "
                "Utiliser obligatoirement quand search_catalog retourne plusieurs postes "
                "avec le même nom mais des num_poste ou nom_affaire différents. "
                "Ne jamais appeler add_to_panier avant que l'utilisateur ait sélectionné."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":     {"type": "string"},
                                "label":  {"type": "string", "description": "Titre court: nom_poste + affaire"},
                                "detail": {"type": "string", "description": "Fournisseur, prix, num_affaire"},
                            },
                            "required": ["id", "label"],
                        },
                    },
                },
                "required": ["question", "options"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_panier",
            "description": (
                "Ajoute des postes confirmés par l'utilisateur au panier du devis. "
                "Appeler uniquement après validation explicite. "
                "IMPORTANT: nom_poste doit être la valeur EXACTE du catalogue (ne pas fusionner avec elements). "
                "nom_affaire est OBLIGATOIRE. Tous les postes doivent avoir le même nom_affaire."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "postes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "nom_poste": {
                                    "type": "string",
                                    "description": "Valeur EXACTE du champ nom_poste du catalogue — ne pas fusionner avec elements",
                                },
                                "nom_affaire": {
                                    "type": "string",
                                    "description": "Valeur EXACTE du champ nom_affaire du catalogue — OBLIGATOIRE",
                                },
                                "num_poste": {
                                    "type": "string",
                                    "description": "Valeur du champ num_poste si disponible dans les résultats",
                                },
                                "ensemble": {"type": "string"},
                                "quantite": {"type": "integer"},
                            },
                            "required": ["nom_poste", "nom_affaire"],
                        },
                        "description": "Liste des postes à ajouter au panier",
                    }
                },
                "required": ["postes"],
            },
        },
    },
]


def _s(value) -> str:
    """Cast any catalog field value to str safely (handles int/float/None)."""
    return str(value).strip() if value is not None else ""


def _query_matches_postes(query: str, results: list[dict]) -> bool:
    """
    True if the query semantically targets a POSTE name (col G) rather than an element.
    Delegates to CatalogAdapter.query_matches_postes which uses the BM25 Snowball tokenizer,
    so plurals like 'orbiteurs' correctly match 'Orbiteur VLM V500'.
    """
    # Fast path: delegate to catalog adapter at call site (needs catalog instance).
    # This standalone version uses a simple regex word intersection as fallback.
    import re as _re
    q_words = _re.findall(r"\w{3,}", query.lower())
    if not q_words:
        return bool(results)
    for row in results:
        nom_poste = _s(row.get("nom_poste")).lower()
        np_words = _re.findall(r"\w+", nom_poste)
        for qw in q_words:
            for npw in np_words:
                if npw.startswith(qw) or qw.startswith(npw):
                    return True
    return False


def _detect_element_conflict(query: str, raw_results: list[dict]) -> dict | None:
    """
    From raw BM25 results, find rows where elements (col H) match the query.

    For each distinct element text, collects all (num_affaire, num_poste) occurrences,
    then computes the intersection to find postes containing ALL requested elements.

    Cases:
    • 1 element  → parent postes (add_poste) + element alone option
    • N elements, common poste(s) → common postes (add_poste) + add separately option
    • N elements, no common poste → one card per element (add_element / show_element_affaires)

    Every option id is JSON with an 'action' field; the frontend dispatches accordingly.
    """
    import re as _re
    from collections import defaultdict

    q_words = _re.findall(r"\w{3,}", query.lower())
    if not q_words:
        return None

    # ── Step 1: group rows by element text ──────────────────────────────────
    element_groups: dict[str, list[dict]] = defaultdict(list)
    for row in raw_results:
        elements_text = _s(row.get("elements"))
        if not elements_text:
            continue
        el_words = _re.findall(r"\w+", elements_text.lower())
        for qw in q_words:
            if any(ew.startswith(qw) or qw.startswith(ew) for ew in el_words):
                element_groups[elements_text].append(row)
                break

    if not element_groups:
        return None

    # ── Step 2: per element, build unique occurrences by (num_affaire, num_poste) ─
    def _occ(row: dict, el_text: str) -> dict:
        price = row.get("fourniture")
        return {
            "action":      "add_element",
            "elements":    el_text,
            "nom_poste":   _s(row.get("nom_poste")),
            "num_poste":   _s(row.get("num_poste")),
            "nom_affaire": _s(row.get("nom_affaire")),
            "num_affaire": _s(row.get("num_affaire")),
            "fournisseur": _s(row.get("fournisseur")),
            "fourniture":  str(price) if price is not None else "",
            "ensemble":    _s(row.get("ensemble")),
        }

    el_data: list[dict] = []
    for el_text, rows in element_groups.items():
        seen: set[tuple] = set()
        occs: list[dict] = []
        for row in rows:
            key = (_s(row.get("num_affaire")), _s(row.get("num_poste")))
            if key not in seen:
                seen.add(key)
                occs.append(_occ(row, el_text))
        el_data.append({"text": el_text, "keys": seen, "occs": occs})

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _poste_option(occ: dict) -> dict:
        detail = f"Affaire : {occ['nom_affaire']}"
        if occ["num_poste"]:
            detail += f" · Poste n°{occ['num_poste']}"
        return {
            "id": json.dumps({
                "action":      "add_poste",
                "nom_poste":   occ["nom_poste"],
                "nom_affaire": occ["nom_affaire"],
                "num_poste":   occ["num_poste"],
            }, ensure_ascii=False),
            "label":  occ["nom_poste"],
            "detail": detail,
        }

    def _alone_option(el: dict) -> dict:
        aff_labels = [o["nom_affaire"] for o in el["occs"] if o["nom_affaire"]]
        detail = ("Affaires : " if len(aff_labels) > 1 else "Affaire : ") + ", ".join(aff_labels) if aff_labels else ""
        if len(el["occs"]) == 1:
            id_data: dict = {**el["occs"][0]}
        else:
            id_data = {"action": "show_element_affaires", "element_text": el["text"], "occurrences": el["occs"]}
        return {"id": json.dumps(id_data, ensure_ascii=False), "label": "Ajouter l'élément seul", "detail": detail}

    # ── Case A: single element ────────────────────────────────────────────────
    if len(el_data) == 1:
        el = el_data[0]
        seen_pk: set[tuple] = set()
        options: list[dict] = []
        for occ in el["occs"]:
            pk = (occ["nom_poste"], occ["nom_affaire"])
            if occ["nom_poste"] and pk not in seen_pk:
                seen_pk.add(pk)
                options.append(_poste_option(occ))
        options.append(_alone_option(el))
        return {
            "type": "element",
            "question": f"L'élément « {el['text']} » appartient à ce(s) poste(s). Que souhaitez-vous ajouter ?",
            "options": options,
        }

    # ── Cases B & C: multiple elements — compute intersection ─────────────────
    common_keys: set[tuple] = el_data[0]["keys"].copy()
    for el in el_data[1:]:
        common_keys &= el["keys"]

    elements_label = ", ".join(f"« {el['text']} »" for el in el_data)

    # ── Case B: common poste(s) found ─────────────────────────────────────────
    if common_keys:
        options = []
        seen_pk = set()
        for num_aff, num_pos in common_keys:
            for el in el_data:
                for occ in el["occs"]:
                    if occ["num_affaire"] == num_aff and occ["num_poste"] == num_pos:
                        pk = (occ["nom_poste"], occ["nom_affaire"])
                        if pk not in seen_pk:
                            seen_pk.add(pk)
                            options.append(_poste_option(occ))
                        break
                else:
                    continue
                break
        sep = [{"text": el["text"], "occurrences": el["occs"]} for el in el_data]
        options.append({
            "id":    json.dumps({"action": "show_elements", "elements": sep}, ensure_ascii=False),
            "label": "Ajouter les éléments séparément",
        })
        return {
            "type": "element",
            "question": f"Les éléments {elements_label} appartiennent au(x) même(s) poste(s). Que souhaitez-vous faire ?",
            "options": options,
        }

    # ── Case C: no common poste — one card per element ────────────────────────
    options = []
    for el in el_data:
        aff_labels = [o["nom_affaire"] for o in el["occs"] if o["nom_affaire"]]
        detail = ("Affaires : " if len(aff_labels) > 1 else "Affaire : ") + ", ".join(aff_labels) if aff_labels else ""
        if len(el["occs"]) == 1:
            id_data = {**el["occs"][0]}
        else:
            id_data = {"action": "show_element_affaires", "element_text": el["text"], "occurrences": el["occs"]}
        options.append({"id": json.dumps(id_data, ensure_ascii=False), "label": el["text"], "detail": detail})
    return {
        "type": "element",
        "question": f"Aucun poste commun pour {elements_label}. Lequel des éléments souhaitez-vous ajouter ?",
        "options": options,
    }


def _build_column_choice(query: str) -> dict:
    """
    Emitted when no nom_poste matches the query.
    Lets the user pick which catalog column to search next.
    """
    return {
        "type": "search_column",
        "question": (
            f"Aucun poste « {query} » trouvé dans le catalogue. "
            "Chercher dans une autre colonne ?"
        ),
        "options": [
            {
                "id": json.dumps(
                    {"action": "search_column", "column": "elements", "query": query},
                    ensure_ascii=False,
                ),
                "label": "Sous-composant",
                "detail": "Chercher parmi les éléments internes (col H)",
            },
            {
                "id": json.dumps(
                    {"action": "search_column", "column": "ensemble", "query": query},
                    ensure_ascii=False,
                ),
                "label": "Type d'ensemble",
                "detail": "Ex : Mécanique, Électrique, Pneumatique",
            },
            {
                "id": json.dumps(
                    {"action": "search_column", "column": "nom_affaire", "query": query},
                    ensure_ascii=False,
                ),
                "label": "Affaire (projet)",
                "detail": "Chercher par nom de projet",
            },
            {
                "id": json.dumps(
                    {"action": "search_column", "column": "fournisseur", "query": query},
                    ensure_ascii=False,
                ),
                "label": "Fournisseur",
                "detail": "Chercher par nom de fournisseur",
            },
            {
                "id": json.dumps({"action": "refine"}, ensure_ascii=False),
                "label": "Affiner ma recherche",
                "detail": "Reformuler votre demande",
            },
        ],
    }


def _build_relevance_choice(query: str, results: list[dict]) -> dict:
    """
    Called when BM25 returned results but none semantically match the query
    (neither postes nor elements). Shows the user three types of actions:
    - One button per found nom_poste (same JSON structure as PRIORITY 1 so the
      frontend can reuse the 'poste' handler).
    - "Search in docs" — tells the LLM to call search_docs.
    - "Refine search" — lets the user rephrase.
    """
    from collections import defaultdict

    # Group by nom_poste and collect occurrences (same logic as PRIORITY 1)
    poste_rows: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        nom_poste = _s(row.get("nom_poste"))
        if nom_poste:
            poste_rows[nom_poste].append(row)

    options: list[dict] = []

    for nom_poste, rows in list(poste_rows.items())[:3]:  # cap at 3 poste options
        seen_occs: set[tuple] = set()
        occurrences: list[dict] = []
        for row in rows:
            occ_key = (_s(row.get("nom_affaire")), _s(row.get("num_poste")))
            if occ_key not in seen_occs:
                seen_occs.add(occ_key)
                occurrences.append({
                    "nom_affaire": _s(row.get("nom_affaire")),
                    "num_poste":   _s(row.get("num_poste")),
                    "fournisseur": _s(row.get("fournisseur")),
                    "ensemble":    _s(row.get("ensemble")),
                })
        row = rows[0]
        detail_parts = []
        if row.get("ensemble"):
            detail_parts.append(f"Ensemble : {row['ensemble']}")
        if row.get("fournisseur"):
            detail_parts.append(f"Fournisseur : {row['fournisseur']}")
        options.append({
            "id": json.dumps(
                {"nom_poste": nom_poste, "occurrences": occurrences},
                ensure_ascii=False,
            ),
            "label": f"Utiliser : {nom_poste}",
            "detail": " · ".join(detail_parts) or None,
        })

    options.append({
        "id": json.dumps({"action": "search_docs", "query": query}, ensure_ascii=False),
        "label": "Chercher dans la documentation PDF",
        "detail": "Rechercher les composants dans les fiches techniques",
    })
    options.append({
        "id": json.dumps({"action": "refine"}, ensure_ascii=False),
        "label": "Affiner la recherche",
        "detail": "Reformuler votre demande",
    })

    return {
        "type": "relevance",
        "question": (
            f"Aucun résultat ne correspond directement à « {query} ». "
            "Que souhaitez-vous faire ?"
        ),
        "options": options,
    }


def _detect_conflict(results: list[dict], current_affaire: str | None) -> dict | None:
    """
    Detect ambiguity in search_catalog results and return a choices event dict, or None.

    Results are already deduplicated by (nom_poste, nom_affaire) before this call,
    so each row represents one distinct poste+affaire combination.

    Priority 1 — Multiple distinct postes: show poste selection cards first.
    Priority 2 — Single poste, multiple affaires: show affaire conflict cards.
    """
    from collections import defaultdict

    if not results:
        return None

    # Group rows by nom_poste first — need distinct_postes before deciding to skip.
    poste_rows: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        nom_poste = _s(row.get("nom_poste"))
        if nom_poste:
            poste_rows[nom_poste].append(row)

    distinct_postes = list(poste_rows.keys())

    # Affaire locked + single unambiguous poste → no conflict to show.
    # Multiple postes always trigger cards (even with a locked affaire) so the
    # user never gets free-form LLM text with potentially wrong names/prices.
    if current_affaire and len(distinct_postes) <= 1:
        return None

    # ── Guard: too many postes — ask user to refine ───────────────────────────
    MAX_POSTE_CARDS = 8
    if len(distinct_postes) > MAX_POSTE_CARDS:
        return {
            "type": "search_column",
            "question": (
                f"La recherche a retourné {len(distinct_postes)} postes différents — "
                "c'est trop pour choisir. Précisez votre recherche ou cherchez dans une autre colonne ?"
            ),
            "options": [
                {
                    "id": json.dumps({"action": "refine"}, ensure_ascii=False),
                    "label": "Affiner ma recherche",
                    "detail": "Reformuler avec un terme plus précis",
                },
                {
                    "id": json.dumps(
                        {"action": "search_column", "column": "elements", "query": ""},
                        ensure_ascii=False,
                    ),
                    "label": "Sous-composant",
                    "detail": "Chercher parmi les éléments internes (col H)",
                },
                {
                    "id": json.dumps(
                        {"action": "search_column", "column": "ensemble", "query": ""},
                        ensure_ascii=False,
                    ),
                    "label": "Type d'ensemble",
                    "detail": "Ex : Mécanique, Électrique, Pneumatique",
                },
                {
                    "id": json.dumps(
                        {"action": "search_column", "column": "nom_affaire", "query": ""},
                        ensure_ascii=False,
                    ),
                    "label": "Affaire (projet)",
                    "detail": "Chercher par nom de projet",
                },
            ],
        }

    # ── PRIORITY 1: Multiple distinct postes ──────────────────────────────────
    # Each card's `id` embeds ALL (nom_affaire, num_poste) occurrences as JSON so
    # the frontend can show affaire sub-cards directly without a new LLM search.
    if len(distinct_postes) > 1:
        options = []
        for nom_poste, rows in poste_rows.items():
            # Collect all unique (nom_affaire, num_poste) occurrences for this poste
            seen_occs: set[tuple] = set()
            occurrences: list[dict] = []
            for row in rows:
                occ_key = (_s(row.get("nom_affaire")), _s(row.get("num_poste")))
                if occ_key not in seen_occs:
                    seen_occs.add(occ_key)
                    occurrences.append({
                        "nom_affaire": _s(row.get("nom_affaire")),
                        "num_poste":   _s(row.get("num_poste")),
                        "fournisseur": _s(row.get("fournisseur")),
                        "ensemble":    _s(row.get("ensemble")),
                    })

            row = rows[0]  # representative row for display
            detail_parts = []
            if row.get("ensemble"):
                detail_parts.append(f"Ensemble : {row['ensemble']}")
            options.append({
                "id": json.dumps(
                    {"nom_poste": nom_poste, "occurrences": occurrences},
                    ensure_ascii=False,
                ),
                "label": nom_poste,
                "detail": " · ".join(detail_parts),
            })
        return {
            "type": "poste",
            "question": "Plusieurs postes correspondent à votre recherche. Lequel souhaitez-vous ajouter ?",
            "options": options,
        }

    # ── PRIORITY 2: Single poste, multiple affaires ───────────────────────────
    if len(distinct_postes) == 1:
        nom_poste = distinct_postes[0]
        rows = poste_rows[nom_poste]
        distinct_affaires = list({
            _s(r.get("nom_affaire"))
            for r in rows
            if r.get("nom_affaire") is not None
        })
        if len(distinct_affaires) > 1:
            seen: set[str] = set()
            options = []
            for row in rows:
                aff = _s(row.get("nom_affaire"))
                if aff and aff not in seen:
                    seen.add(aff)
                    prix_total = row.get("prix_total")
                    detail = f"Poste : {nom_poste}"
                    if prix_total is not None:
                        detail += f" · Prix total : {round(float(prix_total), 2)}€"
                    options.append({
                        "id": aff,
                        "label": aff,
                        "detail": detail,
                    })
            if len(options) > 1:
                return {
                    "type": "affaire",
                    "question": f"Le poste « {nom_poste} » existe dans plusieurs affaires. Quelle affaire utiliser pour ce devis ?",
                    "nom_poste": nom_poste,
                    "options": options,
                }

    return None


class DevisService:
    def __init__(self, catalog_adapter) -> None:
        self.catalog = catalog_adapter

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _ollama_chat(
        self, messages: list[dict], tools: list | None = None
    ) -> dict:
        """Non-streaming Ollama /api/chat call. Used for the tool-calling loop."""
        payload: dict = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,  # Désactive le mode thinking qwen3 pour les appels outils
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
        collection: str,
        conversation_id: str,
    ) -> str:
        """Execute a single tool call and return result as JSON string."""
        if tool_name == "search_catalog":
            query = tool_args.get("query", "")
            logger.info("search_catalog query=%r", query)
            results = await asyncio.to_thread(self.catalog.search, query, 40)
            logger.info("search_catalog returned %d results (raw)", len(results))

            # Fetch current affaire context and user's chosen search scope
            current_affaire = await asyncio.to_thread(
                self.catalog.get_current_affaire, conversation_id
            )
            search_all = await asyncio.to_thread(
                self.catalog.get_search_scope, conversation_id
            )
            if current_affaire and not search_all:
                # Scope restricted to locked affaire (default behaviour)
                filtered = [
                    r for r in results
                    if _s(r.get("nom_affaire")).lower() == current_affaire.lower().strip()
                ]
                if filtered:
                    results = filtered
                    logger.info(
                        "search_catalog filtered to affaire=%r → %d results",
                        current_affaire, len(filtered),
                    )
            elif current_affaire and search_all:
                logger.info(
                    "search_catalog: browse_all mode — catalogue complet retourné (%d résultats)",
                    len(results),
                )

            # Stash the affaire-filtered results BEFORE column filter so element
            # detection in chat_stream always has access to the full raw results.
            self._last_raw_results: list[dict] = results

            # Column filter: restrict results to rows where the requested column
            # matches the query. Default column="nom_poste" means only postes whose
            # name matches the query are returned — elements/ensemble/affaire BM25
            # hits are excluded, avoiding irrelevant poste cards.
            column = tool_args.get("column", "nom_poste")
            col_hits = await asyncio.to_thread(
                self.catalog.filter_to_column_matches, query, column, results
            )
            if col_hits:
                results = col_hits
                logger.info(
                    "search_catalog: column=%r filter → %d/%d résultats retenus",
                    column, len(col_hits), len(self._last_raw_results),
                )
            elif column == "nom_poste":
                # No nom_poste match — BM25 matched only in other columns (elements,
                # ensemble, nom_affaire…). Clear results so chat_stream emits the
                # column choice card instead of showing irrelevant postes.
                results = []
                logger.info(
                    "search_catalog: no nom_poste match for %r — results cleared",
                    query,
                )

            # Extract unique (num_poste, nom_affaire) pairs (normalized) from
            # filtered results, preserving BM25 ranking order.
            seen_pairs: set[tuple] = set()
            pairs: list[tuple[str, str]] = []
            for row in results:
                if not row.get("nom_poste"):
                    continue
                np = str(row.get("num_poste") or "").strip()
                na = (row.get("nom_affaire") or "").strip().lower()
                key = (np, na)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    pairs.append((np, na))

            # SQL GROUP BY + SUM(fourniture) → prix_total per poste+affaire.
            deduped = await asyncio.to_thread(
                self.catalog.get_postes_aggregated, pairs
            )

            # Re-apply BM25 ranking order (SQL GROUP BY doesn't preserve it).
            pair_order = {p: i for i, p in enumerate(pairs)}
            deduped.sort(key=lambda r: pair_order.get(
                (str(r.get("num_poste") or "").strip(),
                 (r.get("nom_affaire") or "").strip().lower()),
                999,
            ))
            logger.info("search_catalog aggregated → %d postes distincts", len(deduped))

            return json.dumps(deduped, ensure_ascii=False, default=str)

        if tool_name == "search_docs":
            query = tool_args.get("query", "")
            # Always use the collection from the request — never trust the LLM's choice
            logger.info("search_docs query=%r collection=%r", query, collection)
            try:
                from core.collection_manager import CollectionManager
                from core.search import RAGEngine

                cm = CollectionManager()
                rag = RAGEngine(
                    nom_collection=collection,
                    prompt_name="defaut",
                    collection_manager=cm,
                )
                result = await asyncio.to_thread(
                    rag.rechercher_debug, query, ""
                )
                chunks = [
                    {
                        "content": c.get("content_preview", c.get("content", "")),
                        "source": c.get("source", ""),
                    }
                    for c in result.get("chunks", [])[:5]
                ]
                logger.info("search_docs returned %d chunks", len(chunks))

                # Enrich result with all catalog poste names so the LLM can
                # cross-reference doc content with known catalog items directly.
                catalog_postes = await asyncio.to_thread(self.catalog.get_all_postes)
                return json.dumps(
                    {"chunks": chunks, "catalog_postes": catalog_postes},
                    ensure_ascii=False,
                )
            except Exception as exc:
                logger.warning("search_docs failed: %s", exc)
                return json.dumps({"error": str(exc)})

        if tool_name == "add_to_panier":
            postes = tool_args.get("postes", [])
            # LLM sometimes sends postes as a JSON string instead of a list
            if isinstance(postes, str):
                try:
                    postes = json.loads(postes)
                except json.JSONDecodeError:
                    postes = []
            # LLM sometimes sends a single dict instead of a list
            if isinstance(postes, dict):
                postes = [postes]
            # If search_all is active (user chose "Tout le catalogue"), update the
            # affaire lock to the new affaire before inserting so the consistency
            # check in add_to_panier passes.
            search_all = await asyncio.to_thread(
                self.catalog.get_search_scope, conversation_id
            )
            if search_all and postes:
                first = postes[0] if isinstance(postes[0], dict) else {}
                new_affaire = (first.get("nom_affaire") or "").strip()
                if new_affaire:
                    await asyncio.to_thread(
                        self.catalog.set_affaire_lock, conversation_id, new_affaire
                    )
            result = await asyncio.to_thread(
                self.catalog.add_to_panier, conversation_id, postes
            )
            added  = result.get("added", [])
            errors = result.get("errors", [])
            current_affaire = result.get("current_affaire")
            response: dict = {
                "added": len(added),
                "postes": [p["nom_poste"] for p in added],
                "current_affaire": current_affaire,
            }
            if errors:
                response["errors"] = errors
                affaire_mismatches = [e for e in errors if e.get("error") == "affaire_mismatch"]
                if affaire_mismatches:
                    response["warning"] = (
                        f"Certains postes ont été rejetés car ils appartiennent à une affaire différente "
                        f"({affaire_mismatches[0]['requested_affaire']}) alors que le devis utilise "
                        f"l'affaire '{current_affaire}'. Informe l'utilisateur."
                    )
            # Reset search scope after a successful add so the next search defaults
            # to the locked affaire again (choice cards will re-ask the user).
            if added:
                await asyncio.to_thread(
                    self.catalog.set_search_scope, conversation_id, False
                )
            return json.dumps(response, ensure_ascii=False, default=str)

        return json.dumps({"error": f"Outil inconnu : {tool_name}"})

    # ── Public API ─────────────────────────────────────────────────────────────

    async def chat_stream(
        self,
        message: str,
        collection: str,
        conversation_id: str,
        history: list[dict],
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator yielding SSE events for a devis chat turn.
        Handles the tool-calling loop then streams the final text response.
        """
        messages: list[dict] = [{"role": "system", "content": _build_system_prompt(collection)}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        _add_to_panier_succeeded = False

        try:
            # ── Tool-calling loop (non-streaming) ─────────────────────────────
            for _ in range(MAX_TOOL_ITERATIONS):
                response = await self._ollama_chat(messages, tools=_TOOLS)
                assistant_msg = response.get("message", {})
                tool_calls = assistant_msg.get("tool_calls") or []

                if not tool_calls:
                    # No more tool calls — ready for final generation
                    break

                # Append assistant tool-call message
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.get("content", ""),
                        "tool_calls": tool_calls,
                    }
                )

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name: str = fn.get("name", "")
                    tool_args = fn.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    logger.info("LLM tool call: %s args=%r", tool_name, tool_args)

                    # ask_user_choice: emit choice cards and stop immediately
                    if tool_name == "ask_user_choice":
                        question = tool_args.get("question", "Choisissez une option :")
                        options  = tool_args.get("options", [])
                        yield {"choices": {"question": question, "options": options}}
                        panier = await asyncio.to_thread(self.catalog.get_panier, conversation_id)
                        yield {"done": True, "panier": panier}
                        return

                    yield {"tool_call": {"name": tool_name, "status": "running"}}
                    result = await self._execute_tool(
                        tool_name, tool_args, collection, conversation_id
                    )

                    # After search_catalog: server-side conflict/element detection
                    # (more reliable than relying on the LLM to call ask_user_choice)
                    if tool_name == "search_catalog":
                        try:
                            current_affaire = await asyncio.to_thread(
                                self.catalog.get_current_affaire, conversation_id
                            )
                            deduped_results = json.loads(result)

                            # Determine whether the query targeted postes or elements.
                            # Use catalog's Snowball tokenizer (handles plurals).
                            matches_postes = await asyncio.to_thread(
                                self.catalog.query_matches_postes,
                                tool_args.get("query", ""),
                                deduped_results,
                            )

                            query_str = tool_args.get("query", "")
                            column = tool_args.get("column", "nom_poste")

                            if column == "elements":
                                # User explicitly chose to search in elements (col H).
                                # _last_raw_results stashed by _execute_tool (unfiltered).
                                raw = getattr(self, "_last_raw_results", [])
                                choice_event = _detect_element_conflict(query_str, raw)
                                if choice_event is None and deduped_results:
                                    choice_event = _build_relevance_choice(
                                        query_str, deduped_results
                                    )

                            elif column != "nom_poste":
                                # ensemble / nom_affaire / fournisseur column search.
                                # Results already filtered to that column in _execute_tool.
                                # Show standard poste conflict cards (PRIORITY 1/2).
                                choice_event = _detect_conflict(deduped_results, current_affaire)

                            elif matches_postes:
                                # Default column="nom_poste" and nom_poste matches found.
                                # PRIORITY 1: multiple postes, PRIORITY 2: single poste multi-affaires.
                                choice_event = _detect_conflict(deduped_results, current_affaire)

                            else:
                                # Default column="nom_poste", no nom_poste match found.
                                # Propose the user to search in another column.
                                logger.info(
                                    "search_catalog: no nom_poste match for %r"
                                    " — emitting column choice card",
                                    query_str,
                                )
                                choice_event = _build_column_choice(query_str)

                        except Exception as exc:
                            logger.warning("conflict/element detection failed: %s", exc)
                            choice_event = None

                        # Enrich choice cards with table data (SQL — exact DB values)
                        if choice_event and choice_event.get("type") == "affaire":
                            nom_poste_ce = choice_event.get("nom_poste", "")
                            for opt in choice_event.get("options", []):
                                try:
                                    elements = await asyncio.to_thread(
                                        self.catalog.get_elements_for_poste,
                                        nom_poste_ce, opt["id"],
                                    )
                                    opt["columns"] = ["Ensemble", "Éléments", "Fournisseur", "Prix €"]
                                    opt["rows"] = [
                                        [
                                            el.get("ensemble") or "—",
                                            el.get("elements") or "—",
                                            el.get("fournisseur") or "—",
                                            str(el.get("fourniture") or "—"),
                                        ]
                                        for el in elements
                                    ]
                                    nb = len(elements)
                                    opt["description"] = f"{nb} élément{'s' if nb > 1 else ''}"
                                except Exception as exc:
                                    logger.warning("enrichment affaire failed: %s", exc)

                        elif choice_event and choice_event.get("type") == "poste":
                            poste_totals: dict[str, list] = {}
                            for row in deduped_results:
                                np = row.get("nom_poste", "")
                                if np:
                                    if np not in poste_totals:
                                        poste_totals[np] = []
                                    poste_totals[np].append((
                                        row.get("nom_affaire", ""),
                                        row.get("prix_total"),
                                    ))
                            for opt in choice_event.get("options", []):
                                try:
                                    parsed = json.loads(opt["id"])
                                    nom_poste_opt = parsed.get("nom_poste", "")
                                    affaires = poste_totals.get(nom_poste_opt, [])
                                    opt["columns"] = ["Affaire", "Prix total"]
                                    opt["rows"] = [
                                        [aff, f"{round(float(pt), 2)}€" if pt is not None else "—"]
                                        for aff, pt in affaires
                                    ]
                                    nb = len(affaires)
                                    opt["description"] = f"{nb} affaire{'s' if nb > 1 else ''}"
                                except Exception as exc:
                                    logger.warning("enrichment poste failed: %s", exc)

                        if choice_event:
                            yield {"tool_call": {"name": tool_name, "status": "done"}}
                            yield {"choices": choice_event}
                            panier = await asyncio.to_thread(
                                self.catalog.get_panier, conversation_id
                            )
                            yield {"done": True, "panier": panier}
                            return

                    # Track add_to_panier success for post-stream choice cards
                    if tool_name == "add_to_panier":
                        try:
                            _r = json.loads(result)
                            if _r.get("added", 0) > 0:
                                _add_to_panier_succeeded = True
                        except Exception:
                            pass

                    # After search_docs: emit a summary of what was found so the
                    # frontend can show the user which sources were consulted.
                    if tool_name == "search_docs":
                        try:
                            parsed = json.loads(result)
                            # result is {"chunks": [...], "catalog_postes": [...]}
                            inner_chunks = (
                                parsed.get("chunks", [])
                                if isinstance(parsed, dict)
                                else parsed
                            )
                            if inner_chunks:
                                sources = list({
                                    c.get("source", "").split("/")[-1]
                                    for c in inner_chunks
                                    if c.get("source")
                                })
                                yield {
                                    "docs_result": {
                                        "query": tool_args.get("query", ""),
                                        "sources": sources[:5],
                                        "count": len(inner_chunks),
                                    }
                                }
                            else:
                                yield {
                                    "docs_result": {
                                        "query": tool_args.get("query", ""),
                                        "sources": [],
                                        "count": 0,
                                    }
                                }
                        except Exception:
                            pass

                    messages.append({"role": "tool", "content": result})
                    yield {"tool_call": {"name": tool_name, "status": "done"}}

                # Emit updated panier after each tool round
                panier = await asyncio.to_thread(
                    self.catalog.get_panier, conversation_id
                )
                yield {"panier": panier}

            # ── Final streaming response (no tools passed) ────────────────────
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": True,
                        "think": False,  # Désactive le mode thinking qwen3
                    },
                ) as resp:
                    _in_think = False  # filtre tokens <think>...</think> (qwen3)
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                if "<think>" in token:
                                    _in_think = True
                                if _in_think:
                                    if "</think>" in token:
                                        _in_think = False
                                        after = token.split("</think>", 1)[1]
                                        if after:
                                            yield {"token": after}
                                else:
                                    yield {"token": token}
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as exc:
            logger.error("Devis chat error: %s", exc, exc_info=True)
            yield {"error": str(exc)}

        finally:
            panier = await asyncio.to_thread(self.catalog.get_panier, conversation_id)
            # Signal the frontend to intercept the next user message with a scope choice.
            # The choice cards are shown client-side before the next query is sent to the LLM.
            yield {"done": True, "panier": panier, "ask_scope": _add_to_panier_succeeded}

    async def generate_devis(
        self, conversation_id: str
    ) -> AsyncGenerator[dict, None]:
        """
        Stream the final devis markdown table from the confirmed panier items.
        Called when the user clicks "Générer le devis".
        """
        panier = await asyncio.to_thread(self.catalog.get_panier, conversation_id)

        if not panier:
            yield {
                "token": (
                    "Le panier est vide. Ajoutez des produits avant de générer le devis."
                )
            }
            yield {"done": True, "panier": []}
            return

        rows = "\n".join(
            f"- Poste: {p['nom_poste']} | Ensemble: {p.get('ensemble') or 'N/A'} | "
            f"Qté: {p.get('quantite', 1)} | "
            f"Fournisseur: {p.get('fournisseur') or 'N/A'} | "
            f"Fourniture: {p.get('fourniture') or 'N/A'} | "
            f"Affaire réf.: {p.get('nom_affaire') or 'N/A'}"
            for p in panier
        )

        prompt = (
            "Génère un devis structuré en tableau Markdown pour les postes ci-dessous.\n"
            "Colonnes exactes : Poste | Ensemble | Qté | Fournisseur | Fourniture | Affaire réf.\n"
            "Ajoute un récapitulatif et des notes si pertinent.\n\n"
            f"Postes confirmés :\n{rows}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant de génération de devis. "
                    "Génère uniquement le tableau Markdown demandé, sans commentaires superflus."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield {"token": token}
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.error("Generate devis error: %s", exc, exc_info=True)
            yield {"error": str(exc)}

        yield {"done": True, "panier": panier}
