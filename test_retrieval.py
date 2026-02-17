#!/usr/bin/env python3
"""
test_retrieval.py — Diagnostic complet du pipeline RAG ChromaDB.

Exécution depuis la racine du projet :
    PYTHONPATH=backend python test_retrieval.py

Ou depuis le répertoire backend/ :
    python ../test_retrieval.py
"""

import sys
from pathlib import Path

# ── Chemins ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# CWD = racine du projet → chroma_db/ est résolu à ./chroma_db/
import os
os.chdir(ROOT)

from core.search import RAGEngine  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────
COLLECTION = "vlm"

# Questions de test — adaptées aux documents réellement présents dans la BDD
# (COMPAQT.pdf, HYMANCO.pdf, Plaquette-FR-GEMINI.pdf, SOLO.pdf,
#  VLM-Robotics-Synthese-Gamme.md)
TEST_CASES = [
    {
        "id": "Q1",
        "question": "Liste les projets liés à la marque SOLO",
        "attente": "Chunks SOLO.pdf + VLM-Robotics-Synthese-Gamme.md avec scores bas",
    },
    {
        "id": "Q2",
        "question": "Quels sont tous les projets et machines disponibles ?",
        "attente": "Résultats variés couvrant les 4 machines (COMPAQT, HYMANCO, GEMINI, SOLO)",
    },
    {
        "id": "Q3",
        "question": "COMPAQT",
        "attente": "COMPAQT.pdf en tête avec score excellent (< 0.30)",
    },
]

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"


def main() -> None:
    print(f"\n{BOLD}{'═'*65}{RESET}")
    print(f"{BOLD}  TEST RETRIEVAL — VLM Robotics RAG{RESET}")
    print(f"{BOLD}{'═'*65}{RESET}")
    print(f"  Collection testée : {COLLECTION}")
    print(f"  Backend path      : {BACKEND}")
    print(f"  CWD               : {Path.cwd()}\n")

    # ── Chargement de la collection ────────────────────────────────────────
    try:
        engine = RAGEngine(COLLECTION, prompt_name="vlm_robotics")
        print(f"  {GREEN}✓ Collection '{COLLECTION}' chargée avec succès.{RESET}\n")
    except Exception as exc:
        print(f"  {RED}✗ Impossible de charger la collection '{COLLECTION}'{RESET}")
        print(f"    Erreur : {exc}")
        print("\n  Vérifications à effectuer :")
        print("    1. Ollama actif          → ollama serve")
        print("    2. Collection existante  → ls chroma_db/")
        print("    3. Documents indexés     → python ingest.py vlm_robotics ./documents/")
        sys.exit(1)

    # ── Lancer les tests ───────────────────────────────────────────────────
    rapports: list[dict] = []

    for cas in TEST_CASES:
        print(f"\n{BOLD}{'─'*65}{RESET}")
        print(f"{BOLD}  {cas['id']} : {cas['question']}{RESET}")
        print(f"  Résultat attendu : {cas['attente']}")
        summary = engine.debug_retrieval(cas["question"], n_results=10)
        rapports.append({"cas": cas, "summary": summary})

    # ── Rapport final ──────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*65}{RESET}")
    print(f"{BOLD}  RAPPORT FINAL{RESET}")
    print(f"{BOLD}{'═'*65}{RESET}\n")

    tous_scores = [s for r in rapports for s in r["summary"]["scores"]]

    for r in rapports:
        s = r["summary"]
        cas = r["cas"]
        score_min_str = f"{s['score_min']:.4f}" if s["score_min"] is not None else "N/A"
        score_moy_str = f"{s['score_moy']:.4f}" if s["score_moy"] is not None else "N/A"

        # Seuils L2 (cos_sim = 1 - L2²/2)
        if s["score_min"] is None:
            couleur = RED
        elif s["score_min"] < 0.45:   # cos_sim > 90%
            couleur = GREEN
        elif s["score_min"] < 0.63:   # cos_sim > 80%
            couleur = "\033[96m"  # cyan
        elif s["score_min"] < 0.85:   # cos_sim > 64%
            couleur = YELLOW
        else:
            couleur = RED

        print(f"  {cas['id']}  {couleur}{score_min_str}{RESET}  ({s['nb_resultats']} résultats, moy {score_moy_str})")
        print(f"       {s['diagnostic']}")
        print()

    # ── Suggestions d'amélioration ─────────────────────────────────────────
    print(f"{BOLD}{'─'*65}{RESET}")
    print(f"{BOLD}  SUGGESTIONS D'AMÉLIORATION{RESET}")
    print(f"{'─'*65}\n")

    if not tous_scores:
        print(f"  {RED}Aucun résultat retourné — vérifiez l'indexation et Ollama.{RESET}\n")
        return

    avg_global = sum(tous_scores) / len(tous_scores)
    min_global = min(tous_scores)

    # Seuils L2 (cos_sim = 1 - L2²/2 pour vecteurs normalisés)
    if min_global > 0.85:
        print(f"  {RED}[CRITIQUE] Scores trop élevés (min L2={min_global:.3f} → cos_sim<64%){RESET}")
        print("  → Problème d'embedding probable :")
        print("    1. Vérifiez EMBEDDING_MODEL dans backend/core/embeddings.py")
        print("    2. Réindexez : python ingest.py vlm_robotics ./documents/ --force")
        print("    3. Vérifiez Ollama : ollama list")

    elif min_global > 0.63:
        print(f"  {YELLOW}[MOYEN] Mismatch sémantique (min L2={min_global:.3f} → cos_sim<80%){RESET}")
        print("  1. Reformulez les questions avec les termes exacts des documents")
        print("  2. Réduire chunk_size 1000 → 500 dans document_manager.py")
        print("  3. Utiliser un filtre metadata : filter={'machine': 'SOLO'}")

    else:
        cos_pct = int((1 - min_global ** 2 / 2) * 100)
        print(f"  {GREEN}[OK] Retrieval fonctionnel (min L2={min_global:.3f} → cos_sim>{cos_pct}%){RESET}")
        print("  Si les réponses LLM sont mauvaises, vérifiez :")
        print("  1. Le prompt — correspond-il au type de question posée ?")
        print("  2. K dynamique — voir _adapter_parametres() dans search.py")

    print(f"\n{BOLD}{'═'*65}{RESET}\n")


if __name__ == "__main__":
    main()
