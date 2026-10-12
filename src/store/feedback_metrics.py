"""Hourly feedback-rate rollup, computed from bot_message_feedback.

This is the only table dashboards query — it stores counts next to their
rates so a daily/weekly figure is SUM(count) / SUM(messages), never an
average of hourly rates.
"""

from src.feedback.sentiment import positive_reactions_sql_array
from src.store import db as database
from src.store import message_feedback

RETENTION_DAYS = 365
# Bounded by bot_message_feedback's own retention — nothing older survives there.
CATCHUP_LIMIT_DAYS = message_feedback.RETENTION_DAYS

# The cause column on bot_reply_labels is deliberately never aggregated here
# (spec decision 10) — no cause rollup, no recompute_cause_buckets, no
# bot_reply_cause_metrics. An ad-hoc GROUP BY cause is the only thing that
# ever reads it; the omission is a decision, not an oversight.
ROLLUP_SQL = """
WITH bounds AS (
    SELECT GREATEST(
        LEAST(
            now() - make_interval(hours => $1),
            COALESCE((SELECT MAX(bucket_start) FROM bot_feedback_metrics), now())
        ),
        now() - make_interval(days => $2)
    ) AS since
),
closed AS (
    SELECT
        date_trunc('hour', feedback.sent_at) AS bucket_start,
        feedback.chat_id,
        feedback.source,
        feedback.trigger,
        COUNT(*)                                          AS messages,
        COUNT(*) FILTER (WHERE feedback.ignored)           AS ignored,
        COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM bot_reply_labels AS label
            WHERE label.chat_id = feedback.chat_id
              AND label.bot_message_id = feedback.message_id
              AND label.reaction = 'MEDIA'
        ))                                                AS media_responded,
        COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM bot_reply_labels AS label
            WHERE label.chat_id = feedback.chat_id
              AND label.bot_message_id = feedback.message_id
              AND label.reaction = 'POSITIVE'
        ))                                                AS positive_responded,
        COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM bot_reply_labels AS label
            WHERE label.chat_id = feedback.chat_id
              AND label.bot_message_id = feedback.message_id
              AND label.reaction = 'NEGATIVE'
        ))                                                AS negative_responded,
        COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM bot_reply_labels AS label
            WHERE label.chat_id = feedback.chat_id
              AND label.bot_message_id = feedback.message_id
              AND label.reaction = 'NEUTRAL'
              AND label.media_type = 'text'
        ))                                                AS neutral_responded,
        COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM bot_reply_labels AS label
            WHERE label.chat_id = feedback.chat_id
              AND label.bot_message_id = feedback.message_id
              AND label.media_type = 'text'
              AND (label.reaction IS NULL OR label.reaction = 'UNCLEAR')
        ))                                                AS unlabelled,
        COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM jsonb_object_keys(feedback.emojis) AS emoji_key
            WHERE replace(emoji_key, U&'\\FE0F', '') = ANY($3::text[])
        ))                                                AS positive_emoji,
        COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM jsonb_object_keys(feedback.emojis) AS emoji_key
            WHERE replace(emoji_key, U&'\\FE0F', '') <> ALL($3::text[])
        ))                                                AS negative_emoji
    FROM bot_message_feedback AS feedback, bounds
    WHERE feedback.ignored IS NOT NULL
      -- Kept in step with src/jobs/reply_labels.py's EXCLUDED_TRIGGERS: a
      -- trigger excluded there has every reply labelled SKIPPED, so without
      -- this clause its bucket would read "N messages, 0% everything" —
      -- a lying zero rather than an honest "not measured".
      AND feedback.trigger <> 'social_link'
      AND feedback.sent_at >= bounds.since
      AND feedback.sent_at < date_trunc('hour', now())
    GROUP BY 1, 2, 3, 4
)
INSERT INTO bot_feedback_metrics
    (bucket_start, chat_id, source, trigger, messages, ignored, ignore_rate,
     media_responded, positive_responded, negative_responded, neutral_responded,
     unlabelled, positive_emoji, negative_emoji,
     media_response_rate, positive_response_rate, negative_response_rate,
     neutral_response_rate, unlabelled_rate, positive_emoji_rate,
     negative_emoji_rate, computed_at)
SELECT
    bucket_start, chat_id, source, trigger, messages, ignored,
    ignored::real / messages,
    media_responded, positive_responded, negative_responded, neutral_responded,
    unlabelled, positive_emoji, negative_emoji,
    media_responded::real / messages,
    positive_responded::real / messages,
    negative_responded::real / messages,
    neutral_responded::real / messages,
    unlabelled::real / messages,
    positive_emoji::real / messages,
    negative_emoji::real / messages,
    now()
FROM closed
ON CONFLICT (bucket_start, chat_id, source, trigger) DO UPDATE SET
    messages = EXCLUDED.messages,
    ignored = EXCLUDED.ignored,
    ignore_rate = EXCLUDED.ignore_rate,
    media_responded = EXCLUDED.media_responded,
    positive_responded = EXCLUDED.positive_responded,
    negative_responded = EXCLUDED.negative_responded,
    neutral_responded = EXCLUDED.neutral_responded,
    unlabelled = EXCLUDED.unlabelled,
    positive_emoji = EXCLUDED.positive_emoji,
    negative_emoji = EXCLUDED.negative_emoji,
    media_response_rate = EXCLUDED.media_response_rate,
    positive_response_rate = EXCLUDED.positive_response_rate,
    negative_response_rate = EXCLUDED.negative_response_rate,
    neutral_response_rate = EXCLUDED.neutral_response_rate,
    unlabelled_rate = EXCLUDED.unlabelled_rate,
    positive_emoji_rate = EXCLUDED.positive_emoji_rate,
    negative_emoji_rate = EXCLUDED.negative_emoji_rate,
    computed_at = EXCLUDED.computed_at
"""


async def recompute_recent_buckets(*, lookback_hours: int = 3) -> int:
    """Recompute every closed hourly bucket touching the recent window.

    "Closed" means every bot_message_feedback row in that hour has ignored
    set (its own 15-minute window has passed) and the hour itself has fully
    elapsed. Recomputing every bucket back to `lookback_hours` (not just the
    newest) on every call means a correction the background classifier
    finishes late still lands in the right bucket, and re-running the job is
    idempotent (ON CONFLICT DO UPDATE).

    After downtime longer than `lookback_hours`, the lower bound instead
    reaches back to the newest bucket already on record, so no hour is
    skipped — bounded by CATCHUP_LIMIT_DAYS, matching how far
    bot_message_feedback itself retains data.

    The positive emoji set is bound as $3 rather than inlined into
    ROLLUP_SQL, so src/feedback/sentiment.py stays the single definition.
    The query's `replace(..., U&'\\FE0F', '')` normalization mirrors
    normalize_reaction, so Postgres and Python agree on emoji like "❤️".

    Args:
        lookback_hours: How far back to always recompute, even with no gap.

    Returns:
        Number of (bucket, chat, source, trigger) rows written or updated.
    """
    async with database.acquire() as conn:
        result = await conn.execute(
            ROLLUP_SQL, lookback_hours, CATCHUP_LIMIT_DAYS, positive_reactions_sql_array(),
        )
    return int(result.split()[-1])


async def cleanup_old() -> int:
    """Delete rows older than RETENTION_DAYS. Returns the number of deleted rows."""
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM bot_feedback_metrics WHERE bucket_start < now() - make_interval(days => $1)",
            RETENTION_DAYS,
        )
    return int(result.split()[-1])
