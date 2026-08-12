"""Add link_material column to unified_messages.

The link-repost feature (docs/superpowers/specs/2026-08-11-link-repost-single-message-design.md)
posts one message per shared link with a compressed caption; the richer
material used to write it (transcript, frame descriptions, comments) was
discarded afterward. This column persists that material on the bot's own
link-repost row so a direct reply to it can be grounded in the same detail
the caption was written from
(docs/superpowers/specs/2026-08-12-link-reply-grounding-design.md). NULL for
every other message — no default needed, TEXT columns default to NULL.

Revision ID: v3_0007
Revises: v3_0006
Create Date: 2026-08-12
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v3_0007"
down_revision: Union[str, None] = "v3_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE unified_messages
        ADD COLUMN link_material TEXT
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
