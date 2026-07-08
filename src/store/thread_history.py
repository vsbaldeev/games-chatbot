"""
Per-thread conversation history for the response LLM.

Each thread corresponds to a Telegram reply chain, keyed by its root message
(``{chat_id}_{root_message_id}``, see :func:`thread_id_for_root`). Flat
(non-reply) exchanges are stored under the prospective chain root — the
triggering message id — so a follow-up reply chain starts pre-seeded with
the exchange that spawned it. Rows are stored oldest-first and retrieved in
that order so the LLM always sees chronological context.

Legacy rows keyed by chat_id alone (the old flat bucket) are no longer read
or written; the retention cleanup ages them out.
"""

import time

from src.store import db as database

HISTORY_RETENTION_DAYS = 60


def thread_id_for_root(chat_id: int, message_id: int) -> str:
    """Build the thread id for the chain rooted at a message.

    Args:
        chat_id: Chat the thread belongs to.
        message_id: Root message of the reply chain (or the prospective
            root — the triggering message — for flat exchanges).

    Returns:
        Thread id string of the form ``{chat_id}_{message_id}``.
    """
    return f"{chat_id}_{message_id}"


async def get_history(*, thread_id: str, limit: int) -> list[dict]:
    """Return up to limit most recent entries for a thread, oldest-first."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content FROM (
                SELECT role, content, created_at
                FROM thread_history
                WHERE thread_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            ) sub
            ORDER BY created_at ASC
            """,
            thread_id, limit,
        )
    return [dict(row) for row in rows]


async def append_turn(
    *, thread_id: str, chat_id: int, human_content: str, ai_content: str
) -> None:
    """Append one human/ai turn. Both rows share the same timestamp prefix."""
    now = time.time()
    async with database.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO thread_history (thread_id, chat_id, role, content, created_at)"
                " VALUES ($1, $2, $3, $4, $5)",
                thread_id, chat_id, "human", human_content, now,
            )
            await conn.execute(
                "INSERT INTO thread_history (thread_id, chat_id, role, content, created_at)"
                " VALUES ($1, $2, $3, $4, $5)",
                thread_id, chat_id, "ai", ai_content, now + 0.001,
            )


async def cleanup_old(*, days: int = HISTORY_RETENTION_DAYS) -> int:
    """Delete rows older than days days. Returns number of deleted rows."""
    cutoff = time.time() - days * 86400
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM thread_history WHERE created_at < $1",
            cutoff,
        )
    return int(result.split()[-1])
