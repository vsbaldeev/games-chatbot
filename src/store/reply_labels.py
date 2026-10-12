"""Per-reply reaction labels: one row per reply to a bot message.

The partial index idx_bot_reply_labels_pending (reaction IS NULL) is the
labelling job's work queue, and fetch_pending is its only read. Backfill and
forward labelling differ only in how many rows are waiting, not in code.

Raw SQL, verified manually per this repo's store-layer convention (see
tests/unit/store/test_unified_messages.py).
"""

from src.store import db as database

MAX_ATTEMPTS = 3
RETENTION_DAYS = 60

PENDING_SQL = """
SELECT
    labels.chat_id,
    labels.reply_message_id,
    labels.bot_message_id,
    labels.media_type,
    labels.attempts,
    COALESCE(reply.content, '')    AS reply_content,
    COALESCE(bot_msg.content, '')  AS bot_content,
    COALESCE(feedback.source, '-') AS source,
    COALESCE(feedback.trigger, '-') AS trigger
FROM bot_reply_labels AS labels
LEFT JOIN unified_messages AS reply
       ON reply.chat_id = labels.chat_id
      AND reply.message_id = labels.reply_message_id
LEFT JOIN unified_messages AS bot_msg
       ON bot_msg.chat_id = labels.chat_id
      AND bot_msg.message_id = labels.bot_message_id
LEFT JOIN bot_message_feedback AS feedback
       ON feedback.chat_id = labels.chat_id
      AND feedback.message_id = labels.bot_message_id
WHERE labels.reaction IS NULL
  AND labels.attempts < $1
ORDER BY labels.created_at
LIMIT $2
"""


async def enqueue(
    *, chat_id: int, reply_message_id: int, bot_message_id: int, depth: int, media_type: str,
) -> None:
    """Record a reply as awaiting a label.

    Idempotent: ON CONFLICT DO NOTHING means a reply enqueued twice (the
    backfill script re-run over a window the live path already covered)
    keeps its first row, attempts and all.

    Args:
        chat_id: Chat the reply was posted in.
        reply_message_id: The member's reply message id.
        bot_message_id: Nearest bot ancestor in the reply chain.
        depth: Hop distance from the reply to that ancestor.
        media_type: The reply's media type, as unified_messages stores it.
    """
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO bot_reply_labels
                (chat_id, reply_message_id, bot_message_id, depth, media_type)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (chat_id, reply_message_id) DO NOTHING
            """,
            chat_id, reply_message_id, bot_message_id, depth, media_type,
        )


async def fetch_pending(*, limit: int) -> list[dict]:
    """Return the oldest unlabelled rows that have attempts left.

    Each row carries everything the labeller needs — the reply text, the bot
    message it answers, and that message's source/trigger — so the job makes
    no further queries per row.

    Args:
        limit: Maximum rows to return; the job's per-run cap.

    Returns:
        A list of dicts, oldest first.
    """
    async with database.acquire() as conn:
        rows = await conn.fetch(PENDING_SQL, MAX_ATTEMPTS, limit)
    return [dict(row) for row in rows]


async def write_label(
    *, chat_id: int, reply_message_id: int, reaction: str, cause: str | None, labelled_by: str,
) -> None:
    """Store a label, leaving any existing one untouched.

    The WHERE clause's `reaction IS NULL` guard means a hand-written label
    (labelled_by = 'human') is never overwritten by a later machine pass, so
    hand-labelled rows survive as a frozen evaluation set.
    """
    async with database.acquire() as conn:
        await conn.execute(
            """
            UPDATE bot_reply_labels
            SET reaction = $3, cause = $4, labelled_by = $5, labelled_at = now()
            WHERE chat_id = $1 AND reply_message_id = $2 AND reaction IS NULL
            """,
            chat_id, reply_message_id, reaction, cause, labelled_by,
        )


async def bump_attempts(*, chat_id: int, reply_message_ids: list[int]) -> None:
    """Count one failed labelling attempt against each of these rows.

    A row that reaches MAX_ATTEMPTS stops being returned by fetch_pending,
    so a reply the model keeps choking on costs three calls, not an endless
    one every fifteen minutes.
    """
    if not reply_message_ids:
        return
    async with database.acquire() as conn:
        await conn.execute(
            """
            UPDATE bot_reply_labels
            SET attempts = attempts + 1
            WHERE chat_id = $1 AND reply_message_id = ANY($2::bigint[])
            """,
            chat_id, reply_message_ids,
        )


async def cleanup_old() -> int:
    """Delete rows older than RETENTION_DAYS. Returns the number of deleted rows."""
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM bot_reply_labels WHERE created_at < now() - make_interval(days => $1)",
            RETENTION_DAYS,
        )
    return int(result.split()[-1])
