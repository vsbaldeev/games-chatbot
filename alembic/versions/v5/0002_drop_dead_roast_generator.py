"""Drop dead roast-generator tables and column.

The roast generator (RoastAgent, Roaster/generate_roast_text) had no live
caller: the /roast command and the weekly scheduled roast were already
retired, and the "offense clap-back" a leftover test comment claimed still
used it does not exist anywhere in the pipeline — insults are handled
entirely by the engagement wind-down engine instead. roast_log and
roast_queue existed only to support that generator (recent-mode avoidance
and round-robin target picking), and user_stats.roasted_count was never
incremented by anything still in the codebase.

Revision ID: v5_0002
Revises: v5_0001
Create Date: 2026-09-16
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v5_0002"
down_revision: Union[str, None] = "v5_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE roast_log")
    op.execute("DROP TABLE roast_queue")
    op.execute("ALTER TABLE user_stats DROP COLUMN roasted_count")


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
