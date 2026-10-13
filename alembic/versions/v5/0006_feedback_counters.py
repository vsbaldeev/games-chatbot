"""Replace the feedback tables with one table of per-message counters.

Feedback counters (docs/superpowers/specs/2026-10-13-feedback-counters-design.md).

Drops bot_feedback_metrics (hourly rollup), bot_reply_labels (per-reply labels
and the labelling queue) and bot_message_feedback, then recreates
bot_message_feedback empty with seven counters and an `ignored` flag that the
application keeps equal to "every counter is 0".

Every row in all three tables is discarded — the operator's explicit call on
2026-10-13: metrics start from zero at deploy.

Revision ID: v5_0006
Revises: v5_0005
Create Date: 2026-10-13
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v5_0006"
down_revision: Union[str, None] = "v5_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE bot_feedback_metrics")
    op.execute("DROP TABLE bot_reply_labels")
    op.execute("DROP TABLE bot_message_feedback")
    op.execute("""
        CREATE TABLE bot_message_feedback (
            chat_id             BIGINT      NOT NULL,
            message_id          BIGINT      NOT NULL,
            source              TEXT        NOT NULL,
            trigger             TEXT        NOT NULL DEFAULT '-',
            filter_verdict      TEXT        NOT NULL DEFAULT '-',
            sent_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
            positive_reactions  INTEGER     NOT NULL DEFAULT 0,
            negative_reactions  INTEGER     NOT NULL DEFAULT 0,
            positive_replies    INTEGER     NOT NULL DEFAULT 0,
            negative_replies    INTEGER     NOT NULL DEFAULT 0,
            neutral_replies     INTEGER     NOT NULL DEFAULT 0,
            media_replies       INTEGER     NOT NULL DEFAULT 0,
            unlabelled_replies  INTEGER     NOT NULL DEFAULT 0,
            ignored             BOOLEAN     NOT NULL DEFAULT TRUE,
            PRIMARY KEY (chat_id, message_id)
        )
    """)
    op.execute("""
        CREATE INDEX idx_bot_message_feedback_sent ON bot_message_feedback (sent_at)
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
