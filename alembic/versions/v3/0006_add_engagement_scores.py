"""Add engagement_scores table for the conversation wind-down engine.

One leaky-bucket attention score per (chat, user): every message the bot has
to deal with adds a weight, and the score decays with a fixed half-life. The
decay is applied lazily in SQL on each write/read, so the row itself is just
the score and the timestamp it was last valid at. Persisted (unlike the old
in-memory insult ladder) so a wound-down user cannot reset their position by
waiting for a redeploy.

Revision ID: v3_0006
Revises: v3_0005
Create Date: 2026-07-15
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v3_0006"
down_revision: Union[str, None] = "v3_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE engagement_scores (
            chat_id        BIGINT           NOT NULL,
            user_id        BIGINT           NOT NULL,
            score          DOUBLE PRECISION NOT NULL,
            last_signal_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (chat_id, user_id)
        )
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
