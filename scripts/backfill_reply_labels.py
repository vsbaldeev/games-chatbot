"""One-off: enqueue historical replies to bot messages for labelling.

Enqueue only — this script labels nothing. The normal 15-minute job
(src/jobs/reply_labels.py) drains the queue under its own per-run cap, so a
backfill of thousands of replies spreads over nights instead of arriving as
one burst.

Idempotent: enqueue is ON CONFLICT DO NOTHING on (chat_id, reply_message_id),
so an interrupted run can simply be repeated.

Only the nearest bot ancestor is enqueued per reply, matching track_reply:
the reply must point directly at a message that is in bot_message_feedback.
Deeper chains are left to the live path, which is why the backfill undercounts
multi-hop threads from before deployment. Accepted — it is a one-time edge.

Usage:
    python -m scripts.backfill_reply_labels --days 60
    python -m scripts.backfill_reply_labels --days 60 --dry-run
"""

import argparse
import asyncio

from src import log
from src.store import db as database
from src.store import reply_labels

logger = log.get_logger(__name__)

# unified_messages.created_at is a Unix epoch DOUBLE PRECISION (see
# alembic/versions/v3/0001_initial_schema.py and unified_messages.cleanup_old,
# which compares it against time.time() - days * 86400), not a TIMESTAMPTZ like
# bot_reply_labels.created_at. The cutoff is therefore converted to an epoch
# float with EXTRACT(EPOCH FROM ...); comparing the column directly against a
# timestamptz expression has no Postgres operator and errors on every run.
HISTORICAL_REPLIES_SQL = """
SELECT
    reply.chat_id,
    reply.message_id   AS reply_message_id,
    reply.reply_to_msg_id AS bot_message_id,
    reply.media_type
FROM unified_messages AS reply
JOIN bot_message_feedback AS feedback
  ON feedback.chat_id = reply.chat_id
 AND feedback.message_id = reply.reply_to_msg_id
LEFT JOIN bot_reply_labels AS existing
  ON existing.chat_id = reply.chat_id
 AND existing.reply_message_id = reply.message_id
WHERE reply.reply_to_msg_id IS NOT NULL
  AND existing.reply_message_id IS NULL
  AND reply.created_at >= EXTRACT(EPOCH FROM (now() - make_interval(days => $1)))::double precision
ORDER BY reply.created_at
"""


async def find_historical_replies(days: int) -> list[dict]:
    """Return replies pointing at a tracked bot message and not yet enqueued."""
    async with database.acquire() as conn:
        rows = await conn.fetch(HISTORICAL_REPLIES_SQL, days)
    return [dict(row) for row in rows]


async def backfill(days: int, dry_run: bool) -> int:
    """Enqueue every historical reply found. Returns the number enqueued."""
    rows = await find_historical_replies(days)
    logger.info("Backfill: found %d replies to enqueue over %d days", len(rows), days)
    if dry_run:
        return 0
    for row in rows:
        await reply_labels.enqueue(
            chat_id=row["chat_id"],
            reply_message_id=row["reply_message_id"],
            bot_message_id=row["bot_message_id"],
            depth=1,
            media_type=row["media_type"],
        )
    logger.info("Backfill: enqueued %d replies", len(rows))
    return len(rows)


async def main() -> None:
    """Parse arguments, open the pool, run the backfill, close the pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=60, help="How far back to look")
    parser.add_argument("--dry-run", action="store_true", help="Count only, enqueue nothing")
    arguments = parser.parse_args()
    await database.init()
    try:
        await backfill(arguments.days, arguments.dry_run)
    finally:
        await database.close()


if __name__ == "__main__":
    asyncio.run(main())
