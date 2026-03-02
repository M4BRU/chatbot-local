"""Conversation CRUD endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.dependencies import get_conversation_manager
from backend.core.conversation_manager import ConversationManager

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class CreateConversationBody(BaseModel):
    title: str
    mode: str = "chat"


class AddMessageBody(BaseModel):
    role: str
    content: str


class UpdateTitleBody(BaseModel):
    title: str


@router.get("")
async def list_conversations(cm: ConversationManager = Depends(get_conversation_manager)):
    return await asyncio.to_thread(cm.list_conversations)


@router.post("", status_code=201)
async def create_conversation(
    body: CreateConversationBody,
    cm: ConversationManager = Depends(get_conversation_manager),
):
    return await asyncio.to_thread(cm.create_conversation, body.title, body.mode)


@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    cm: ConversationManager = Depends(get_conversation_manager),
):
    return await asyncio.to_thread(cm.get_messages, conversation_id)


@router.post("/{conversation_id}/messages", status_code=201)
async def add_message(
    conversation_id: str,
    body: AddMessageBody,
    cm: ConversationManager = Depends(get_conversation_manager),
):
    return await asyncio.to_thread(cm.add_message, conversation_id, body.role, body.content)


@router.patch("/{conversation_id}")
async def update_title(
    conversation_id: str,
    body: UpdateTitleBody,
    cm: ConversationManager = Depends(get_conversation_manager),
):
    await asyncio.to_thread(cm.update_title, conversation_id, body.title)
    return {"ok": True}


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    cm: ConversationManager = Depends(get_conversation_manager),
):
    await asyncio.to_thread(cm.delete_conversation, conversation_id)
