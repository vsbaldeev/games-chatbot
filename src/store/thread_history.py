"""
Per-thread conversation history for the response LLM.

Each thread corresponds to a Telegram reply chain (keyed by its root message)
or the flat chat (keyed by chat_id alone).  Rows are stored oldest-first and
retrieved in that order so the LLM always sees chronological context.
"""

import time

from src.store import db as database

HISTORY_RETENTION_DAYS = 60


async def init_table() -> None:
    async with database.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_history (
                    thread_id  TEXT             NOT NULL,
                    chat_id    BIGINT           NOT NULL,
                    role       TEXT             NOT NULL,
                    content    TEXT             NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thread_history_lookup
                ON thread_history (thread_id, created_at)
            """)


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
