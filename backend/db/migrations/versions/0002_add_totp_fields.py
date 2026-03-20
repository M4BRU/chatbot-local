"""add_totp_fields

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-18

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("totp_secret", sa.String(length=64), nullable=True))
    op.add_column(
        "users",
        sa.Column("totp_enabled", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("users", "totp_enabled")
    op.drop_column("users", "totp_secret")
