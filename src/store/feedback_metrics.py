"""Hourly feedback-rate rollup, computed from bot_message_feedback.

This is the only table dashboards query — it stores counts next to their
rates so a daily/weekly figure is SUM(count) / SUM(messages), never an
average of hourly rates.
"""

from src.store import db as database
from src.store import message_feedback

RETENTION_DAYS = 365
# Bounded by bot_message_feedback's own retention — nothing older survives there.
CATCHUP_LIMIT_DAYS = message_feedback.RETENTION_DAYS

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
        COUNT(*) FILTER (WHERE feedback.corrections > 0)   AS corrected,
        COUNT(*) FILTER (WHERE feedback.reactions > 0)     AS reacted,
        COUNT(*) FILTER (WHERE feedback.replies > 0)       AS replied,
        AVG(feedback.thread_depth)                         AS avg_thread_depth,
        percentile_cont(0.5) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (feedback.first_feedback_at - feedback.sent_at))
        ) FILTER (WHERE feedback.first_feedback_at IS NOT NULL) AS median_first_feedback_s
    FROM bot_message_feedback AS feedback, bounds
    WHERE feedback.ignored IS NOT NULL
      AND feedback.sent_at >= bounds.since
      AND feedback.sent_at < date_trunc('hour', now())
    GROUP BY 1, 2, 3, 4
)
INSERT INTO bot_feedback_metrics
    (bucket_start, chat_id, source, trigger, messages, ignored, corrected, reacted, replied,
     ignore_rate, correction_rate, reaction_rate, reply_rate, avg_thread_depth,
     median_first_feedback_s, computed_at)
SELECT
    bucket_start, chat_id, source, trigger, messages, ignored, corrected, reacted, replied,
    ignored::real / messages, corrected::real / messages,
    reacted::real / messages, replied::real / messages,
    avg_thread_depth, median_first_feedback_s, now()
FROM closed
ON CONFLICT (bucket_start, chat_id, source, trigger) DO UPDATE SET
    messages = EXCLUDED.messages,
    ignored = EXCLUDED.ignored,
    corrected = EXCLUDED.corrected,
    reacted = EXCLUDED.reacted,
    replied = EXCLUDED.replied,
    ignore_rate = EXCLUDED.ignore_rate,
    correction_rate = EXCLUDED.correction_rate,
    reaction_rate = EXCLUDED.reaction_rate,
    reply_rate = EXCLUDED.reply_rate,
    avg_thread_depth = EXCLUDED.avg_thread_depth,
    median_first_feedback_s = EXCLUDED.median_first_feedback_s,
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

    Args:
        lookback_hours: How far back to always recompute, even with no gap.

    Returns:
        Number of (bucket, chat, source, trigger) rows written or updated.
    """
    async with database.acquire() as conn:
        result = await conn.execute(ROLLUP_SQL, lookback_hours, CATCHUP_LIMIT_DAYS)
    return int(result.split()[-1])


async def cleanup_old() -> int:
    """Delete rows older than RETENTION_DAYS. Returns the number of deleted rows."""
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM bot_feedback_metrics WHERE bucket_start < now() - make_interval(days => $1)",
            RETENTION_DAYS,
        )
    return int(result.split()[-1])
