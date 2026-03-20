"""Conversation CRUD endpoints — scoped par user_id (SQLAlchemy async)."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.adapters.auth_adapter import get_current_user
from backend.db.base import get_async_session
from backend.db.models import Conversation, Message, UserTable

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CreateConversationBody(BaseModel):
    title: str
    mode: str = "chat"


class AddMessageBody(BaseModel):
    role: str
    content: str


class UpdateTitleBody(BaseModel):
    title: str


@router.get("")
async def list_conversations(
    current_user: UserTable = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    result = await session.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    convs = result.scalars().all()
    return [
        {
            "id": str(c.id),
            "title": c.title,
            "mode": c.mode,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat(),
        }
        for c in convs
    ]


@router.post("", status_code=201)
async def create_conversation(
    body: CreateConversationBody,
    current_user: UserTable = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    conv = Conversation(
        user_id=current_user.id,
        title=body.title,
        mode=body.mode,
    )
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return {
        "id": str(conv.id),
        "title": conv.title,
        "mode": conv.mode,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
    }


@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    current_user: UserTable = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    conv = await _get_owned_conversation(conversation_id, current_user.id, session)
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc())
    )
    msgs = result.scalars().all()
    return [
        {
            "id": str(m.id),
            "conversation_id": str(m.conversation_id),
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
        }
        for m in msgs
    ]


@router.post("/{conversation_id}/messages", status_code=201)
async def add_message(
    conversation_id: str,
    body: AddMessageBody,
    current_user: UserTable = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    conv = await _get_owned_conversation(conversation_id, current_user.id, session)
    msg = Message(
        conversation_id=conv.id,
        role=body.role,
        content=body.content,
    )
    session.add(msg)
    conv.updated_at = _now()
    await session.commit()
    await session.refresh(msg)
    return {
        "id": str(msg.id),
        "conversation_id": str(msg.conversation_id),
        "role": msg.role,
        "content": msg.content,
        "created_at": msg.created_at.isoformat(),
    }


@router.patch("/{conversation_id}")
async def update_title(
    conversation_id: str,
    body: UpdateTitleBody,
    current_user: UserTable = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    conv = await _get_owned_conversation(conversation_id, current_user.id, session)
    conv.title = body.title
    conv.updated_at = _now()
    await session.commit()
    return {"ok": True}


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    current_user: UserTable = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    conv = await _get_owned_conversation(conversation_id, current_user.id, session)
    await session.delete(conv)
    await session.commit()


async def _get_owned_conversation(
    conversation_id: str,
    user_id: uuid.UUID,
    session: AsyncSession,
) -> Conversation:
    """Récupère une conversation et vérifie qu'elle appartient à l'utilisateur."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation introuvable")

    result = await session.execute(
        select(Conversation).where(
            Conversation.id == conv_uuid,
            Conversation.user_id == user_id,
        )
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation introuvable")
    return conv
