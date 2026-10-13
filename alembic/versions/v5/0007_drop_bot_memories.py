"""Drop the bot_memories table left over from the retired Жора persona.

bot_memories held the persona's invented life canon (episodes, facts and the
daily activity phrase). The scheduled life posts were retired on 2026-08-10
and nothing has read or written the table since src/store/bot_memories.py was
deleted; the persona itself is now removed, so the data is dropped with it.
The table's indexes go with it.

Every row is discarded — the operator's explicit call on 2026-10-13.

Revision ID: v5_0007
Revises: v5_0006
Create Date: 2026-10-13
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v5_0007"
down_revision: Union[str, None] = "v5_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE bot_memories")


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
