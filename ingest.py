"""
ingest.py — CLI d'indexation multi-collections.

Usage :
    python ingest.py <collection> <chemin> [--force]

Exemples :
    python ingest.py vlm_robotics ./documents/
    python ingest.py vlm_robotics ./documents/SOLO.pdf --force
"""

import argparse
import sys
import time
from pathlib import Path

from core.embeddings import verifier_ollama
from core.parsers import extensions_supportees
from core.collection_manager import CollectionManager
from core.document_manager import DocumentManager


def main():
    parser = argparse.ArgumentParser(
        description="Indexation de documents dans une collection ChromaDB."
    )
    parser.add_argument("collection", help="Nom de la collection cible")
    parser.add_argument("chemin", help="Fichier ou dossier à indexer")
    parser.add_argument("--force", action="store_true", help="Ré-indexer même si déjà présent")
    args = parser.parse_args()

    print("=" * 60)
    print(f"   Indexation dans la collection : {args.collection}")
    print("=" * 60)
    print()

    # Vérification Ollama
    if not verifier_ollama():
        print("Erreur : Ollama n'est pas accessible sur http://localhost:11434")
        print("Lancez-le avec : ollama serve")
        sys.exit(1)
    print("Ollama est accessible.\n")

    chemin = Path(args.chemin)
    if not chemin.exists():
        print(f"Erreur : {chemin} n'existe pas.")
        sys.exit(1)

    # Lister les fichiers à indexer
    ext_ok = set(extensions_supportees())
    if chemin.is_file():
        if chemin.suffix.lower() not in ext_ok:
            print(f"Erreur : format {chemin.suffix} non supporté.")
            print(f"Formats acceptés : {', '.join(sorted(ext_ok))}")
            sys.exit(1)
        fichiers = [chemin]
    else:
        fichiers = sorted(
            f for f in chemin.iterdir()
            if f.is_file() and f.suffix.lower() in ext_ok
        )

    if not fichiers:
        print(f"Aucun fichier supporté trouvé dans {chemin}")
        print(f"Formats acceptés : {', '.join(sorted(ext_ok))}")
        sys.exit(1)

    print(f"{len(fichiers)} fichier(s) trouvé(s)\n")

    # Indexation
    cm = CollectionManager()
    dm = DocumentManager(cm)
    debut = time.time()
    total_chunks = 0
    nb_indexes = 0
    nb_ignores = 0

    for i, fichier in enumerate(fichiers, start=1):
        print(f"  [{i}/{len(fichiers)}] {fichier.name}...", end=" ", flush=True)
        resultat = dm.ajouter_document(args.collection, fichier, force=args.force)
        print(f"-> {resultat['message']}")

        if resultat["status"] == "indexed":
            nb_indexes += 1
            total_chunks += resultat["chunks"]
        else:
            nb_ignores += 1

    duree = time.time() - debut
    print()
    print("=" * 60)
    print(f"   {nb_indexes} document(s) indexé(s), {nb_ignores} ignoré(s)")
    print(f"   {total_chunks} chunks créés au total")
    print(f"   Durée : {duree:.1f} secondes")
    print(f"   Collection : {args.collection}")
    print("=" * 60)


if __name__ == "__main__":
    main()
