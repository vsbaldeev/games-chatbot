"""Drop dead reaction-tracking tables and columns.

roast_reactions and message_reaction_counts were written by handlers that
have since been removed (the emoji-reaction MessageReactionHandler, and an
apply_reaction_counts helper that had no caller at all), and had no reader
of their own — get_anchor_stats, the one function that read roast_reactions,
was itself never called. user_stats.laugh_reactions / heart_reactions /
fire_reactions / thumbsup_reactions were populated by that same removed
handler and are dropped for the same reason.

Revision ID: v5_0001
Revises: v3_0008
Create Date: 2026-09-16
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v5_0001"
down_revision: Union[str, None] = "v3_0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE roast_reactions")
    op.execute("DROP TABLE message_reaction_counts")
    op.execute("""
        ALTER TABLE user_stats
        DROP COLUMN laugh_reactions,
        DROP COLUMN heart_reactions,
        DROP COLUMN fire_reactions,
        DROP COLUMN thumbsup_reactions
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
