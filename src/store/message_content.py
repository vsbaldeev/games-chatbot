"""Per-message content facts for the bot's pipeline messages.

One row per bot message that answers a pipeline turn, written by register()
at send time: why the message exists (content_kind) and how the worker stage
ended (worker_status). response_label is filled later by a background
classifier, only for free-form explicit replies. Rows are never updated
otherwise, and are joined to bot_message_feedback on (chat_id, message_id).
"""

from src.store import db as database

RETENTION_DAYS = 60


async def register(
    *, chat_id: int, message_id: int, content_kind: str, worker_status: str | None,
) -> None:
    """Record what kind of message the bot sent and how its worker stage ended.

    Idempotent: ON CONFLICT DO NOTHING means a message registered twice (a
    search-notification message later edited into its final text) keeps its
    first row.

    Args:
        chat_id: Chat the message was sent in.
        message_id: The sent message's id.
        content_kind: A ContentKind value (src.pipeline.state).
        worker_status: A WorkerStatus value, or None when the worker stage
            never finished (canned early exits, aborted pipeline).
    """
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO bot_message_content (chat_id, message_id, content_kind, worker_status)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (chat_id, message_id) DO NOTHING
            """,
            chat_id, message_id, content_kind, worker_status,
        )


async def set_response_label(*, chat_id: int, message_id: int, label: str) -> None:
    """Store the classifier's verdict on a message. A no-op for a message with no row."""
    async with database.acquire() as conn:
        await conn.execute(
            """
            UPDATE bot_message_content
            SET response_label = $3
            WHERE chat_id = $1 AND message_id = $2
            """,
            chat_id, message_id, label,
        )


async def cleanup_old() -> int:
    """Delete rows older than RETENTION_DAYS. Returns the number of deleted rows."""
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM bot_message_content WHERE sent_at < now() - make_interval(days => $1)",
            RETENTION_DAYS,
        )
    return int(result.split()[-1])
