"""Add sticker_descriptions cache table.

Sticker rows in unified_messages used to keep the [sticker] placeholder
forever — there was no vision-enrichment path for them. Descriptions are now
produced lazily and cached here keyed by Telegram's file_unique_id, which is
stable across resends and bots, so each distinct sticker costs exactly one
vision call regardless of how often it is resent.

Revision ID: v3_0003
Revises: v3_0002
Create Date: 2026-07-07
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v3_0003"
down_revision: Union[str, None] = "v3_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE sticker_descriptions (
            file_unique_id TEXT             PRIMARY KEY,
            description    TEXT             NOT NULL,
            created_at     DOUBLE PRECISION NOT NULL
        )
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
