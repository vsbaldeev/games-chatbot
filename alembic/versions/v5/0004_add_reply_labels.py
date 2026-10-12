"""Add bot_reply_labels, add the reply-reaction metric columns, drop the dead ones.

Reply reaction metrics (docs/superpowers/specs/2026-10-09-reply-reaction-metrics-design.md).

Four parts:
  1. bot_reply_labels — one row per reply to a bot message. The partial index
     on (created_at) WHERE reaction IS NULL is the labelling job's work queue
     and its only query.
  2. Fourteen additive columns on bot_feedback_metrics — the new count and
     rate columns this feature reports.
  3. Eight columns are dropped outright: corrected, correction_rate, reacted,
     reaction_rate, replied, reply_rate, avg_thread_depth and
     median_first_feedback_s. Corrections go because correction tracking is
     dropped; reacted/reaction_rate and replied/reply_rate because "any
     reaction" and "any reply" are not questions anyone reads once the signal
     is split by sentiment; the last two because they are off-subject.

     Dropped rather than left to linger, on the operator's explicit call: a
     table that reads clearly matters more here than three and a half weeks of
     a metric nobody reads. ROLLUP_SQL ends up writing messages, ignored,
     ignore_rate and the fourteen new columns — seventeen in all, every one
     answering a question that was actually asked.
  4. Four columns are dropped from bot_message_feedback: corrections,
     reactions, replies, thread_depth. Each was read only by one of the eight
     rollup columns above, so after part 3 the bot would write them and nothing
     would ever read them. emojis stays — it feeds the two emoji rates — and
     first_feedback_at stays because close_expired_windows derives `ignored`
     from it, which ignore_rate depends on. The table goes from 13 columns
     to 8.

The new metric columns are nullable with no default for the same reason — rows
written before this migration have no honest value for them.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "v5_0004"
down_revision: Union[str, None] = "v5_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE bot_reply_labels (
            chat_id          BIGINT      NOT NULL,
            reply_message_id BIGINT      NOT NULL,
            bot_message_id   BIGINT      NOT NULL,
            depth            INTEGER     NOT NULL,
            media_type       TEXT        NOT NULL,
            reaction         TEXT,
            cause            TEXT,
            labelled_by      TEXT,
            attempts         INTEGER     NOT NULL DEFAULT 0,
            labelled_at      TIMESTAMPTZ,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (chat_id, reply_message_id)
        )
    """)
    op.execute("""
        CREATE INDEX idx_bot_reply_labels_pending
            ON bot_reply_labels (created_at) WHERE reaction IS NULL
    """)
    op.execute("""
        CREATE INDEX idx_bot_reply_labels_created ON bot_reply_labels (created_at)
    """)
    for column in (
        "media_responded INTEGER",
        "positive_emoji INTEGER",
        "negative_emoji INTEGER",
        "positive_responded INTEGER",
        "negative_responded INTEGER",
        "neutral_responded INTEGER",
        "unlabelled INTEGER",
        "media_response_rate REAL",
        "positive_emoji_rate REAL",
        "negative_emoji_rate REAL",
        "positive_response_rate REAL",
        "negative_response_rate REAL",
        "neutral_response_rate REAL",
        "unlabelled_rate REAL",
    ):
        op.execute(f"ALTER TABLE bot_feedback_metrics ADD COLUMN {column}")
    # Dropped outright rather than left to linger: a table nobody can read at
    # a glance is worse than a table missing three weeks of a metric nobody
    # reads. bot_feedback_metrics only started filling on 2026-09-17, so this
    # discards roughly three and a half weeks of these eight — irreversible,
    # and the operator's explicit call on 2026-10-12.
    #
    # messages, ignored and ignore_rate stay: ignore_rate is one of the seven
    # metrics, and it is computed from feedback.ignored rather than from
    # reacted or replied, so dropping those two does not touch it.
    for column in (
        "corrected", "correction_rate",
        "reacted", "reaction_rate",
        "replied", "reply_rate",
        "avg_thread_depth", "median_first_feedback_s",
    ):
        op.execute(f"ALTER TABLE bot_feedback_metrics DROP COLUMN {column}")
    # Cascade: with those eight gone, these four per-message columns are
    # written by the bot and read by nothing. emojis stays (it feeds the two
    # emoji rates), and first_feedback_at stays because close_expired_windows
    # derives `ignored` from it — ignore_rate depends on that.
    for column in ("corrections", "reactions", "replies", "thread_depth"):
        op.execute(f"ALTER TABLE bot_message_feedback DROP COLUMN {column}")


def downgrade() -> None:
    raise NotImplementedError("Forward-only migrations: roll back with a new forward migration")
