"""SQLite-backed conversation persistence manager."""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path("/app/documents/conversations.db")


class ConversationManager:
    """Manages conversations and messages in a local SQLite database."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def list_conversations(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def create_conversation(self, title: str, mode: str = "chat") -> dict:
        now = self._now()
        conv_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, mode, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conv_id, title, mode, now, now),
            )
        return {"id": conv_id, "title": title, "mode": mode, "created_at": now, "updated_at": now}

    def get_messages(self, conversation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message(self, conversation_id: str, role: str, content: str) -> dict:
        now = self._now()
        msg_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (msg_id, conversation_id, role, content, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return {"id": msg_id, "conversation_id": conversation_id, "role": role, "content": content, "created_at": now}

    def update_title(self, conversation_id: str, title: str) -> None:
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
