"""Add 'activity' kind to bot_memories for the silent daily activity refresh.

``activity`` rows are a lightweight daily-invented "what is Жора doing right
now" phrase, generated without posting to chat: no embedding, no
``post_format``, the phrase stored in both ``content`` and
``current_activity``. They share the table with ``episode``/``fact`` rows
for the same reason those two do — a single ``get_current_activity`` query
needs to see the newest phrase regardless of which kind produced it.

Revision ID: v3_0005
Revises: v3_0004
Create Date: 2026-07-11
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v3_0005"
down_revision: Union[str, None] = "v3_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE bot_memories DROP CONSTRAINT bot_memories_kind_check")
    op.execute("""
        ALTER TABLE bot_memories
        ADD CONSTRAINT bot_memories_kind_check
        CHECK (kind IN ('episode', 'fact', 'activity'))
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
