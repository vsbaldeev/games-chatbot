"""Add bot_memories table for the persona's own life canon.

Two row kinds share one table: ``episode`` rows are full posted life-story
entries (used for narrative continuity when writing the next episode);
``fact`` rows are short durable canon sentences injected into chat replies.
Facts survive episode pruning, which is why both kinds live together rather
than in separate tables. Mirrors ``user_memories`` for embeddings and the
HNSW index.

Revision ID: v3_0004
Revises: v3_0003
Create Date: 2026-07-10
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v3_0004"
down_revision: Union[str, None] = "v3_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE bot_memories (
            id                BIGSERIAL        PRIMARY KEY,
            kind              TEXT             NOT NULL CHECK (kind IN ('episode', 'fact')),
            content           TEXT             NOT NULL,
            embedding         vector(384),
            post_format       TEXT,
            posted_at         DOUBLE PRECISION,
            current_activity  TEXT,
            created_at        DOUBLE PRECISION NOT NULL,
            updated_at        DOUBLE PRECISION NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX idx_bot_memories_kind_time
        ON bot_memories (kind, updated_at DESC)
    """)
    op.execute("""
        CREATE INDEX idx_bot_memories_hnsw
        ON bot_memories USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
