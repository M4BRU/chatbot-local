"""
Tous les prompts et schemas JSON pour le mode devis.
Centralisé ici pour faciliter les modifications et tests.
"""

import re

# ── JSON schemas pour Ollama grammar-constrained generation ───────────────────
# format=schema → le LLM retourne TOUJOURS du JSON valide (élimine les parse failures).

SCHEMA_DIMENSIONS = {
    "type": "object",
    "properties": {
        "dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string"},
                    "query":     {"type": "string"},
                },
                "required": ["dimension", "query"],
            },
        }
    },
    "required": ["dimensions"],
}

SCHEMA_COMPONENTS = {
    "type": "object",
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nom":   {"type": "string"},
                    "specs": {"type": "string"},
                },
                "required": ["nom", "specs"],
            },
        }
    },
    "required": ["components"],
}

# ── Tool definitions (Ollama format) ──────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan_tasks",
            "description": "Enregistre la liste des actions à effectuer. Chaque item = label court.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ex: [\"search vireur\", \"update orbiteur atelier=5\"]",
                    }
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_panier_item",
            "description": "Modifie un poste existant dans le panier (jours ou option).",
            "parameters": {
                "type": "object",
                "properties": {
                    "nom_poste": {
                        "type": "string",
                        "description": "Nom EXACT du poste (copier depuis PANIER ACTUEL)",
                    },
                    "nbre_jours_etude":   {"type": "integer", "description": "Jours d'étude"},
                    "nbre_jours_atelier": {"type": "integer", "description": "Jours d'atelier"},
                    "nbre_jours_client":  {"type": "integer", "description": "Jours client"},
                    "is_option": {"type": "boolean", "description": "true = option"},
                },
                "required": ["nom_poste"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Recherche dans le catalogue VLM. Retourne des postes avec prix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Terme de recherche",
                    },
                    "column": {
                        "type": "string",
                        "enum": ["nom_poste", "elements", "ensemble", "nom_affaire", "fournisseur"],
                        "description": "Colonne cible (défaut: nom_poste)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_collection_excel",
            "description": "Recherche dans les Excel de la collection (tarifs, tableaux). Pas le catalogue VLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question ou terme de recherche",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_findings",
            "description": "Liste les composants trouvés dans les docs. Appeler après search_docs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Noms spécifiques de modèles/références trouvés",
                    },
                    "context": {
                        "type": "string",
                        "description": "Résumé court de ce qui a été trouvé",
                    },
                    "catalog_matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "nom_poste": {"type": "string"},
                                "spec_status": {
                                    "type": "string",
                                    "enum": ["match", "partial", "no_match", "unknown"],
                                },
                                "note": {"type": "string"},
                            },
                            "required": ["nom_poste", "spec_status"],
                        },
                        "description": "Statut specs par poste catalogue",
                    },
                    "doc_only_models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Modèles dans les docs mais absents du catalogue",
                    },
                    "suggestion": {
                        "type": "string",
                        "description": "Recommandation courte",
                    },
                },
                "required": ["components"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Recherche dans les PDFs techniques. Retourne composants + specs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Type de composant + contraintes (ex: 'vireur 8T basculeur')",
                    },
                    "catalog_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Noms exacts des postes catalogue à vérifier dans les docs",
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
            "description": "Présente des choix cliquables à l'utilisateur.",
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
                                "label":  {"type": "string"},
                                "detail": {"type": "string"},
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
            "name": "set_devis_settings",
            "description": "Met à jour les coefficients du devis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coefficient": {
                        "type": "number",
                        "description": "Coefficient fournitures en %",
                    },
                    "coef_final": {
                        "type": "number",
                        "description": "Coefficient final en %",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_panier",
            "description": "Ajoute des postes au panier. Utiliser les valeurs EXACTES du catalogue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "postes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "nom_poste":   {"type": "string", "description": "Valeur EXACTE de nom_poste"},
                                "nom_affaire": {"type": "string", "description": "Valeur EXACTE de nom_affaire"},
                                "num_poste":   {"type": "string"},
                                "ensemble":    {"type": "string"},
                                "quantite":    {"type": "integer"},
                                "nbre_jours_etude":   {"type": "integer"},
                                "nbre_jours_atelier": {"type": "integer"},
                                "nbre_jours_client":  {"type": "integer"},
                                "is_option":          {"type": "boolean"},
                            },
                            "required": ["nom_poste", "nom_affaire"],
                        },
                    }
                },
                "required": ["postes"],
            },
        },
    },
]

# Soft harness extra tools
TOOLS_SOFT_EXTRA = [
    {
        "type": "function",
        "function": {
            "name": "write_todos",
            "description": "Écrit ou met à jour la liste de tâches de planification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":    {"type": "string"},
                                "label": {"type": "string"},
                                "done":  {"type": "boolean"},
                            },
                            "required": ["id", "label"],
                        },
                    }
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_todos",
            "description": "Lit la liste de tâches courante.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def build_system_prompt(
    collection: str,
    current_coefficient: float = 0.0,
    current_coef_final: float = 0.0,
    panier: list[dict] | None = None,
    task_list: list[dict] | None = None,
    rfq_context: str | None = None,
) -> str:
    if panier:
        lines = [
            f"  • {p['nom_poste']} | affaire : {p.get('nom_affaire') or '?'} | "
            f"Étude:{p.get('nbre_jours_etude', 0)}j Atelier:{p.get('nbre_jours_atelier', 0)}j Client:{p.get('nbre_jours_client', 0)}j"
            f"{' | EN OPTION' if p.get('is_option') else ''}"
            for p in panier
        ]
        panier_section = "Postes déjà dans le devis :\n" + "\n".join(lines)
    else:
        panier_section = "Le panier est vide."

    if task_list:
        panier_noms_lower = {(p.get("nom_poste") or "").lower() for p in (panier or [])}

        def _task_done(task: dict) -> bool:
            if task.get("done"):
                return True
            q_words = set(re.findall(r"\w{3,}", task["query"].lower()))
            return any(
                bool(q_words & set(re.findall(r"\w{3,}", nom)))
                for nom in panier_noms_lower
            )

        task_lines = [
            f"  {'[✓]' if _task_done(t) else '[ ]'} {t['query']}"
            for t in task_list
        ]
        task_section = "LISTE DE TÂCHES :\n" + "\n".join(task_lines)
        pending = [t for t in task_list if not _task_done(t)]
        next_task = f"\n  → Prochain item à chercher : \"{pending[0]['query']}\"" if pending else "\n  → Tous les items sont ajoutés."
        task_section += next_task
    else:
        task_section = "LISTE DE TÂCHES : (vide — utilise plan_tasks si plusieurs postes demandés)"

    rfq_section = (
        f"\nCONTEXTE RFQ (analyse documentaire préalable — LECTURE SEULE) :\n{rfq_context}\n"
        "↳ Ces informations viennent des documents PDF/Excel. Elles NE contiennent PAS de num_poste catalogue.\n"
        "↳ Pour ajouter un composant identifié ici : search_catalog obligatoire d'abord.\n"
        if rfq_context
        else ""
    )

    return f"""Tu es un assistant de devis pour VLM Robotics.

CATALOGUE : historique de projets réels. Chaque poste (nom_poste) a un num_poste et appartient à une nom_affaire.

PANIER ACTUEL :
{panier_section}

{task_section}
{rfq_section}COEFFICIENTS : fournitures={current_coefficient}% · final={current_coef_final}%

━━━ QUE FAIRE ━━━

ÉTAPE 1 — Toujours commencer par un tool call, jamais du texte.
• Demande contient "Sélectionné :" → add_to_panier directement.
• Tâche [ ] en attente → search_catalog ou update_panier_item selon la tâche.
• Message "[SYSTÈME]" ou "Cherche X" → search_catalog(query=X).
• Sinon → search_catalog pour ce que l'utilisateur demande.

ÉTAPE 2 — Après search_catalog :
• Champ "_hint" présent → add_to_panier immédiatement.
• Champ "_next_action" présent → suis l'instruction du champ (tool call obligatoire, pas de texte).
• Plusieurs postes retournés, pas de _hint → ask_user_choice avec les postes trouvés.

ÉTAPE 3 — Après search_docs :
• Appelle report_findings avec les noms de composants du résultat. Pas de texte entre les deux.

RÉPONSE TEXTE : 1 phrase max. Jamais de code, jamais de liste inventée.

━━━ RÈGLES ━━━
• nom_poste, nom_affaire, num_poste : copier les valeurs EXACTES des résultats search_catalog.
• "en option" → is_option=true. Jours mentionnés → nbre_jours_etude/atelier/client.
• Données tabulaires (tarifs, tableaux Excel) → search_collection_excel.
• set_devis_settings uniquement si l'utilisateur demande à changer les coefficients.
• UNE tâche à la fois."""
