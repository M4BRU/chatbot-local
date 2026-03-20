"""Configuration fastapi-users — UserManager, transports JWT, backends auth."""

import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend,
    CookieTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import Settings
from backend.db.base import get_async_session
from backend.db.models import UserTable

_settings = Settings()


# ── Schémas Pydantic ──────────────────────────────────────────────────────────

from fastapi_users import schemas


class UserRead(schemas.BaseUser[uuid.UUID]):
    role: str
    totp_enabled: bool = False


class UserCreate(schemas.BaseUserCreate):
    role: str = "STANDARD"


class UserUpdate(schemas.BaseUserUpdate):
    role: str | None = None


# ── Database adapter ──────────────────────────────────────────────────────────

async def get_user_db(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[SQLAlchemyUserDatabase, None]:
    yield SQLAlchemyUserDatabase(session, UserTable)


# ── UserManager ───────────────────────────────────────────────────────────────

SECRET = _settings.jwt_secret_key


class UserManager(UUIDIDMixin, BaseUserManager[UserTable, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),
) -> AsyncGenerator[UserManager, None]:
    yield UserManager(user_db)


# ── Transport : cookie httpOnly ───────────────────────────────────────────────

cookie_transport = CookieTransport(
    cookie_name="access_token",
    cookie_max_age=_settings.jwt_access_token_expire_minutes * 60,
    cookie_secure=_settings.cookie_secure,
    cookie_samesite=_settings.cookie_samesite,  # type: ignore[arg-type]
    cookie_httponly=True,
)


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(
        secret=_settings.jwt_secret_key,
        lifetime_seconds=_settings.jwt_access_token_expire_minutes * 60,
        algorithm=_settings.jwt_algorithm,
    )


auth_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# ── FastAPIUsers instance ─────────────────────────────────────────────────────

fastapi_users = FastAPIUsers[UserTable, uuid.UUID](
    get_user_manager,
    [auth_backend],
)

# Dependency shortcuts
get_current_user = fastapi_users.current_user(active=True)
get_current_superuser = fastapi_users.current_user(active=True, superuser=True)
