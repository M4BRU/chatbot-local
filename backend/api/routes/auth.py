"""Routes d'authentification — login custom (avec interception TOTP), logout, register, /me."""

import datetime
import uuid
from typing import Annotated

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from fastapi_users.exceptions import UserAlreadyExists

from backend.adapters.auth_adapter import (
    UserCreate,
    UserManager,
    UserRead,
    UserUpdate,
    auth_backend,
    fastapi_users,
    get_user_manager,
)
from backend.config.settings import Settings
from backend.db.models import UserTable

_settings = Settings()

router = APIRouter(tags=["auth"])


# ── Helpers pre_auth_token ────────────────────────────────────────────────────


def _create_pre_auth_token(user_id: uuid.UUID) -> str:
    """JWT court-vécu (5 min) signalant que le TOTP reste à valider."""
    payload = {
        "sub": str(user_id),
        "totp_pending": True,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
    }
    return pyjwt.encode(payload, _settings.jwt_secret_key, algorithm=_settings.jwt_algorithm)


def _decode_pre_auth_token(token: str) -> uuid.UUID:
    """Décode et valide un pre_auth_token, retourne l'user_id."""
    try:
        payload = pyjwt.decode(
            token, _settings.jwt_secret_key, algorithms=[_settings.jwt_algorithm]
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="pre_auth_token expiré"
        )
    except pyjwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="pre_auth_token invalide"
        )
    if not payload.get("totp_pending"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token ne correspond pas à une session TOTP",
        )
    return uuid.UUID(payload["sub"])


# ── Login custom (remplace fastapi-users /auth/jwt/login) ────────────────────


@router.post("/auth/jwt/login")
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    user_manager: UserManager = Depends(get_user_manager),
):
    """Login en 1 ou 2 étapes selon que le TOTP est activé ou non."""
    user = await user_manager.authenticate(form_data)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LOGIN_BAD_CREDENTIALS",
        )

    # TOTP activé → retourner un pre_auth_token (étape 1 sur 2)
    if getattr(user, "totp_enabled", False):
        pre_auth_token = _create_pre_auth_token(user.id)
        return {"totp_required": True, "pre_auth_token": pre_auth_token}

    # Pas de TOTP → générer le cookie directement (comme fastapi-users)
    strategy = auth_backend.get_strategy()
    token = await strategy.write_token(user)
    return await auth_backend.transport.get_login_response(token)


# ── Logout ───────────────────────────────────────────────────────────────────


@router.post("/auth/jwt/logout")
async def logout():
    """Efface le cookie access_token."""
    return await auth_backend.transport.get_logout_response()


# ── Register, /users/me — routeurs fastapi-users inchangés ───────────────────

# POST /auth/register
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
)

# GET  /auth/users/me  → profil courant (appelé par le middleware Next.js)
# PATCH /auth/users/me → modifier son profil
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/auth/users",
)


# ── Gestion utilisateurs — ADMIN et DEV uniquement ───────────────────────────

_ADMIN_ROLES = {"ADMIN", "DEV"}
_get_current_user = fastapi_users.current_user(active=True)


def _require_admin_or_dev(current_user: UserTable = Depends(_get_current_user)) -> UserTable:
    if current_user.role not in _ADMIN_ROLES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accès réservé")
    return current_user


# GET /api/v1/users — liste tous les utilisateurs
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.base import get_async_session


@router.get("/api/v1/users", response_model=list[UserRead])
async def list_users(
    _: UserTable = Depends(_require_admin_or_dev),
    session: AsyncSession = Depends(get_async_session),
):
    result = await session.execute(select(UserTable).order_by(UserTable.created_at))
    return result.scalars().all()


# POST /api/v1/users — crée un utilisateur avec rôle choisi
@router.post("/api/v1/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    _: UserTable = Depends(_require_admin_or_dev),
    user_manager: UserManager = Depends(get_user_manager),
):
    try:
        user = await user_manager.create(body, safe=False)
    except UserAlreadyExists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Un utilisateur avec cet email existe déjà",
        )
    return user
