"""Routes TOTP — setup, enable, disable, verify."""

import base64
import uuid
from io import BytesIO

import pyotp
import qrcode
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.adapters.auth_adapter import (
    UserManager,
    auth_backend,
    fastapi_users,
    get_user_manager,
)
from backend.api.routes.auth import _decode_pre_auth_token
from backend.db.base import get_async_session
from backend.db.models import UserTable

router = APIRouter(tags=["totp"])

_get_current_user = fastapi_users.current_user(active=True)


# ── Schémas ──────────────────────────────────────────────────────────────────


class TotpEnableRequest(BaseModel):
    code: str


class TotpDisableRequest(BaseModel):
    code: str


class TotpVerifyRequest(BaseModel):
    pre_auth_token: str
    code: str


# ── GET /auth/totp/setup ──────────────────────────────────────────────────────


@router.get("/auth/totp/setup")
async def totp_setup(
    user: UserTable = Depends(_get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    """Génère un nouveau secret TOTP, le persiste (non activé) et retourne le QR code."""
    secret = pyotp.random_base32()

    user.totp_secret = secret
    session.add(user)
    await session.commit()

    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(name=user.email, issuer_name="Chatbot VLM")

    # QR code en PNG base64
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    qr_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "secret": secret,
        "otpauth_uri": uri,
        "qr_data_url": f"data:image/png;base64,{qr_b64}",
    }


# ── POST /auth/totp/enable ────────────────────────────────────────────────────


@router.post("/auth/totp/enable")
async def totp_enable(
    body: TotpEnableRequest,
    user: UserTable = Depends(_get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    """Active le TOTP après vérification du code fourni par l'application 2FA."""
    if not user.totp_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Appelez d'abord GET /auth/totp/setup",
        )

    totp = pyotp.TOTP(user.totp_secret)
    if not totp.verify(body.code, valid_window=1):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Code TOTP invalide")

    user.totp_enabled = True
    session.add(user)
    await session.commit()

    return {"detail": "TOTP activé avec succès"}


# ── POST /auth/totp/disable ───────────────────────────────────────────────────


@router.post("/auth/totp/disable")
async def totp_disable(
    body: TotpDisableRequest,
    user: UserTable = Depends(_get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    """Désactive le TOTP après confirmation par code (évite la désactivation accidentelle)."""
    if not user.totp_enabled or not user.totp_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="TOTP non activé"
        )

    totp = pyotp.TOTP(user.totp_secret)
    if not totp.verify(body.code, valid_window=1):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Code TOTP invalide")

    user.totp_enabled = False
    user.totp_secret = None
    session.add(user)
    await session.commit()

    return {"detail": "TOTP désactivé"}


# ── POST /auth/totp/verify ────────────────────────────────────────────────────


@router.post("/auth/totp/verify")
async def totp_verify(
    body: TotpVerifyRequest,
    user_manager: UserManager = Depends(get_user_manager),
):
    """Étape 2 du login TOTP : valide le code et pose le cookie d'authentification."""
    user_id = _decode_pre_auth_token(body.pre_auth_token)

    try:
        from fastapi_users.exceptions import UserNotExists
        user = await user_manager.get(user_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Utilisateur invalide"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Utilisateur inactif"
        )

    if not getattr(user, "totp_enabled", False) or not getattr(user, "totp_secret", None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="TOTP non configuré"
        )

    totp = pyotp.TOTP(user.totp_secret)
    if not totp.verify(body.code, valid_window=1):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Code TOTP invalide")

    strategy = auth_backend.get_strategy()
    token = await strategy.write_token(user)
    return await auth_backend.transport.get_login_response(token)
