"""Add bot feedback and LLM-trace tables.

Three new tables back online tracking of implicit user feedback on bot
messages (docs/superpowers/specs/2026-09-17-bot-feedback-metrics-design.md):

- llm_system_prompts: the response system prompt deduplicated by content
  hash, so bot_llm_log rows store only a reference to it.
- bot_llm_log: one row per delivered pipeline reply — the exact prompt and
  response, for inspecting corrected replies later.
- bot_message_feedback: one row per bot message of any source — reaction/
  reply/correction counts gathered only within the first 15 minutes.
- bot_feedback_metrics: hourly rollup of bot_message_feedback, the only
  table dashboards query.

Timestamps use TIMESTAMPTZ (not the DOUBLE PRECISION epoch older tables
use) so rows read naturally in psql/Grafana — a deliberate departure from
house convention for this feature.

Revision ID: v5_0003
Revises: v5_0002
Create Date: 2026-09-17
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v5_0003"
down_revision: Union[str, None] = "v5_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE llm_system_prompts (
            sha256        TEXT        PRIMARY KEY,
            content       TEXT        NOT NULL,
            first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE bot_llm_log (
            chat_id            BIGINT      NOT NULL,
            bot_message_id     BIGINT      NOT NULL,
            user_message_id    BIGINT      NOT NULL,
            user_id            BIGINT      NOT NULL,
            trigger            TEXT        NOT NULL,
            filter_verdict     TEXT        NOT NULL DEFAULT '-',
            system_prompt_sha  TEXT        NOT NULL REFERENCES llm_system_prompts(sha256),
            history_messages   JSONB       NOT NULL,
            user_prompt        TEXT        NOT NULL,
            raw_response       TEXT        NOT NULL,
            language_corrected BOOLEAN     NOT NULL DEFAULT FALSE,
            response           TEXT        NOT NULL,
            model              TEXT,
            input_tokens       INTEGER,
            output_tokens      INTEGER,
            latency_ms         INTEGER     NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (chat_id, bot_message_id)
        )
    """)
    op.execute("""
        CREATE INDEX idx_bot_llm_log_created ON bot_llm_log (created_at)
    """)
    op.execute("""
        CREATE TABLE bot_message_feedback (
            chat_id           BIGINT      NOT NULL,
            message_id        BIGINT      NOT NULL,
            source            TEXT        NOT NULL,
            trigger           TEXT        NOT NULL DEFAULT '-',
            filter_verdict    TEXT        NOT NULL DEFAULT '-',
            sent_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            reactions         INTEGER     NOT NULL DEFAULT 0,
            emojis            JSONB       NOT NULL DEFAULT '{}'::jsonb,
            replies           INTEGER     NOT NULL DEFAULT 0,
            corrections       INTEGER     NOT NULL DEFAULT 0,
            thread_depth      INTEGER     NOT NULL DEFAULT 0,
            first_feedback_at TIMESTAMPTZ,
            ignored           BOOLEAN,
            PRIMARY KEY (chat_id, message_id)
        )
    """)
    op.execute("""
        CREATE INDEX idx_bot_message_feedback_sent ON bot_message_feedback (sent_at)
    """)
    op.execute("""
        CREATE TABLE bot_feedback_metrics (
            bucket_start             TIMESTAMPTZ NOT NULL,
            chat_id                  BIGINT      NOT NULL,
            source                   TEXT        NOT NULL,
            trigger                  TEXT        NOT NULL,
            messages                 INTEGER     NOT NULL,
            ignored                  INTEGER     NOT NULL,
            corrected                INTEGER     NOT NULL,
            reacted                  INTEGER     NOT NULL,
            replied                  INTEGER     NOT NULL,
            ignore_rate              REAL        NOT NULL,
            correction_rate          REAL        NOT NULL,
            reaction_rate            REAL        NOT NULL,
            reply_rate               REAL        NOT NULL,
            avg_thread_depth         REAL        NOT NULL,
            median_first_feedback_s  REAL,
            computed_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bucket_start, chat_id, source, trigger)
        )
    """)


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
