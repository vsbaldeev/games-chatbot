"""Add is_forwarded flag to unified_messages.

Forwarded messages used to be indistinguishable from a participant's own words
once persisted, so LLM prompts attributed forwarded channel content to the
person who forwarded it. The flag lets conversation rendering mark such rows
explicitly (``[переслал]``).

Revision ID: v3_0002
Revises: v3_0001
Create Date: 2026-07-06
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v3_0002"
down_revision: Union[str, None] = "v3_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE unified_messages
        ADD COLUMN is_forwarded BOOLEAN NOT NULL DEFAULT FALSE
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
