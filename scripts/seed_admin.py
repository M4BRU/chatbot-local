#!/usr/bin/env python3
"""
Seed admin user in the database.

Usage:
  docker compose exec backend python scripts/seed_admin.py \
    --email admin@vlm.local \
    --password "ChangeMe123!" \
    --role ADMIN
"""

import argparse
import asyncio
import os
import sys

# Ajoute la racine du projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main(email: str, password: str, role: str) -> None:
    from backend.adapters.auth_adapter import get_user_db, get_user_manager
    from backend.adapters.auth_adapter import UserCreate
    from backend.db.base import AsyncSessionLocal
    from fastapi_users.exceptions import UserAlreadyExists

    async with AsyncSessionLocal() as session:
        # Crée le contexte de dépendance manuellement
        user_db_gen = get_user_db(session)
        user_db = await user_db_gen.__anext__()

        user_manager_gen = get_user_manager(user_db)
        user_manager = await user_manager_gen.__anext__()

        try:
            user = await user_manager.create(
                UserCreate(email=email, password=password, role=role),
                safe=False,
            )
            print(f"[OK] Utilisateur créé : {user.email} (role={user.role}, id={user.id})")
        except UserAlreadyExists:
            print(f"[INFO] Utilisateur déjà existant : {email}")
        except Exception as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed admin user")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--role", default="ADMIN", choices=["ADMIN", "COMMERCIAL", "STANDARD"])
    args = parser.parse_args()

    asyncio.run(main(args.email, args.password, args.role))
