#!/usr/bin/env python3
"""
Migration script : collections Qdrant séparées → collection unifiée avec filtre collection_id.

Contexte RBAC : chaque collection Qdrant existante devient un segment dans une collection
unifiée. Le champ `collection_id` dans le payload Qdrant permet de filtrer par collection
lors de la recherche, tout en supportant la recherche multi-collection (ADMIN / COMMERCIAL).

Usage :
  # Depuis l'hôte (nécessite qdrant-client installé) :
  python scripts/migrate_qdrant_unified.py --qdrant-url http://localhost:6333

  # Depuis le backend container :
  docker compose exec backend python /app/scripts/migrate_qdrant_unified.py

Options :
  --qdrant-url   URL Qdrant (défaut: http://localhost:6333)
  --unified-name Nom de la collection unifiée (défaut: unified)
  --dry-run      Affiche ce qui serait fait sans modifier Qdrant
"""

import argparse
import sys

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct, VectorParams, Distance
except ImportError:
    print("[ERROR] qdrant-client non installé. Lancez : pip install qdrant-client")
    sys.exit(1)


def migrate(qdrant_url: str, unified_name: str, dry_run: bool) -> None:
    client = QdrantClient(url=qdrant_url)

    collections_resp = client.get_collections()
    collection_names = [c.name for c in collections_resp.collections]

    if not collection_names:
        print("[INFO] Aucune collection Qdrant trouvée.")
        return

    print(f"[INFO] Collections trouvées : {collection_names}")
    to_migrate = [n for n in collection_names if n != unified_name]

    if not to_migrate:
        print("[INFO] Rien à migrer.")
        return

    # Récupère la config vectorielle de la première collection source
    first = client.get_collection(to_migrate[0])
    vectors_config = first.config.params.vectors

    if unified_name not in collection_names:
        if dry_run:
            print(f"[DRY-RUN] Créerait la collection unifiée '{unified_name}'")
        else:
            client.create_collection(
                collection_name=unified_name,
                vectors_config=vectors_config,
            )
            print(f"[OK] Collection '{unified_name}' créée.")

    total_migrated = 0
    for coll_name in to_migrate:
        offset = None
        batch_count = 0
        while True:
            scroll_result = client.scroll(
                collection_name=coll_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points, next_offset = scroll_result

            if not points:
                break

            new_points = []
            for p in points:
                payload = dict(p.payload or {})
                # Injecte collection_id si absent
                if "collection_id" not in payload:
                    payload["collection_id"] = coll_name
                new_points.append(PointStruct(id=p.id, vector=p.vector, payload=payload))

            if dry_run:
                print(f"[DRY-RUN] {coll_name}: upsertrait {len(new_points)} points (batch {batch_count + 1})")
            else:
                client.upsert(collection_name=unified_name, points=new_points)
                batch_count += 1

            total_migrated += len(points)

            if next_offset is None:
                break
            offset = next_offset

        print(f"[OK] '{coll_name}' : {total_migrated} points migrés → '{unified_name}'")

    if not dry_run:
        print(f"\n[DONE] Migration terminée. Total : {total_migrated} points dans '{unified_name}'.")
        print("[INFO] Les collections sources sont conservées — supprimer manuellement si souhaité.")
        print(f"[INFO] Mettre à jour rbac.yaml pour référencer '{unified_name}' + VECTOR_DB=qdrant.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Qdrant collections to unified collection")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--unified-name", default="unified")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    migrate(args.qdrant_url, args.unified_name, args.dry_run)
