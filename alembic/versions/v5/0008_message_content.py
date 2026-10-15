"""Add bot_message_content: what the bot put into the chat, per message.

One row per bot message that answers a pipeline turn, written at send time
(docs/superpowers/specs/2026-10-14-message-content-metrics-design.md).
Content facts live apart from bot_message_feedback because they are written
once, while feedback counters keep moving; the two join on
(chat_id, message_id). No foreign key — each table has its own retention.

Revision ID: v5_0008
Revises: v5_0007
Create Date: 2026-10-14
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v5_0008"
down_revision: Union[str, None] = "v5_0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE bot_message_content (
            chat_id         BIGINT      NOT NULL,
            message_id      BIGINT      NOT NULL,
            content_kind    TEXT        NOT NULL CHECK (content_kind IN (
                'response', 'link_summary', 'wind_down_reply', 'insult_reply',
                'prompt_injection_reply', 'limit_notice', 'context_limit_notice',
                'cap_notice', 'media_failure_notice', 'error'
            )),
            worker_status   TEXT        CHECK (worker_status IN (
                'ok', 'skipped', 'context_overflow', 'failed'
            )),
            response_label  TEXT        CHECK (response_label IN (
                'ANSWERED', 'REFUSED', 'NON_ANSWER', 'UNCLEAR'
            )),
            sent_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (chat_id, message_id)
        )
    """)
    op.execute("""
        CREATE INDEX idx_bot_message_content_sent ON bot_message_content (sent_at)
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
