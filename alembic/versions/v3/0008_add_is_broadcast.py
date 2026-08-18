"""Add is_broadcast column to unified_messages.

Marks the bot's own group-wide announcements (weekly roles, group profiles)
so the router's addressing gate can tell them apart from ordinary bot
messages. A member replying to a broadcast is usually talking to the chat
about it, not to the bot, so those replies are allowed a NOT_ADDRESSED
verdict; every other bot message keeps the blanket "any reply addresses me"
rule. Same role link_material already plays for link-repost rows.

Revision ID: v3_0008
Revises: v3_0007
Create Date: 2026-08-18
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v3_0008"
down_revision: Union[str, None] = "v3_0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE unified_messages
        ADD COLUMN is_broadcast BOOLEAN NOT NULL DEFAULT FALSE
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
