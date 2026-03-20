"""AuthorizationService — contrôle d'accès RBAC aux collections RAG."""

from pathlib import Path

import yaml
from fastapi import HTTPException, status
from pydantic import BaseModel, ValidationError


class RbacConfig(BaseModel):
    """Schéma de validation du fichier rbac.yaml."""
    roles: dict[str, list[str]]


class AuthorizationService:
    """
    Charge la config RBAC depuis un fichier YAML et expose les méthodes
    get_authorized_collections() et assert_can_access().

    Comportement default-deny : si le rôle est inconnu ou la config absente,
    aucune collection n'est autorisée et toute assertion lève une 403.

    Raises RuntimeError au démarrage si le fichier est absent ou invalide.
    """

    def __init__(self, config_path: str = "/app/config/rbac.yaml") -> None:
        path = Path(config_path)
        if not path.exists():
            raise RuntimeError(
                f"RBAC config introuvable : {config_path}. "
                "Vérifier le volume mount docker-compose ou créer le fichier."
            )
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            config = RbacConfig(**raw)
        except (yaml.YAMLError, ValidationError, TypeError) as exc:
            raise RuntimeError(
                f"RBAC config invalide ({config_path}) : {exc}"
            ) from exc

        self._role_collections: dict[str, list[str]] = config.roles

    def is_superuser(self, role: str) -> bool:
        """Retourne True si le rôle a accès à toutes les collections (wildcard '*')."""
        return "*" in self._role_collections.get(role, [])

    def get_authorized_collections(self, role: str) -> list[str]:
        """Retourne la liste des collections autorisées pour ce rôle.

        Si le rôle contient '*', retourne ['*'] (wildcard — caller doit tester is_superuser).
        Default-deny : retourne [] si le rôle est inconnu.
        """
        return self._role_collections.get(role, [])

    def assert_can_access(self, role: str, collection_id: str) -> None:
        """Lève HTTP 403 si le rôle n'a pas accès à la collection.

        Default-deny : rôle inconnu = accès refusé partout.
        Wildcard '*' : accès à toutes les collections.
        """
        authorized = self.get_authorized_collections(role)
        if not collection_id or (collection_id not in authorized and "*" not in authorized):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accès refusé à la collection '{collection_id}' pour le rôle '{role}'.",
            )
