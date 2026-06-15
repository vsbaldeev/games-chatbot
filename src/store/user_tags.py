"""
Persistent store for weekly member roles ("tags") and the reason each was
assigned.

The weekly roles job invents one short role per active chat member and records
it here keyed by (chat_id, user_id), together with a short justification. The
chat pipeline reads it back so the bot can explain a member's role on request.
"""

import time

from src.store import db as database


async def init_table() -> None:
    """Create the user_tags table if it does not already exist."""
    async with database.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_tags (
                chat_id     BIGINT           NOT NULL,
                user_id     BIGINT           NOT NULL,
                tag         TEXT             NOT NULL,
                reason      TEXT             NOT NULL,
                assigned_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (chat_id, user_id)
            )
        """)


async def get_tag(*, chat_id: int, user_id: int) -> dict | None:
    """Return the member's current role and reason, or None when unassigned.

    Args:
        chat_id: Group chat the role belongs to.
        user_id: Telegram user whose role is requested.

    Returns:
        A dict with ``tag`` and ``reason`` keys, or ``None`` if the member has
        no stored role.
    """
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT tag, reason FROM user_tags WHERE chat_id = $1 AND user_id = $2",
            chat_id, user_id,
        )
    if row is None:
        return None
    return {"tag": row["tag"], "reason": row["reason"]}


async def get_latest_assignment_time() -> float | None:
    """Return the newest ``assigned_at`` across all stored roles, or None.

    Used by the startup catch-up to tell whether the most recent scheduled
    roles run actually produced any tags.

    Returns:
        The maximum ``assigned_at`` epoch timestamp over every row, or ``None``
        when no roles have ever been assigned.
    """
    async with database.acquire() as conn:
        latest = await conn.fetchval("SELECT MAX(assigned_at) FROM user_tags")
    return latest


async def upsert_tags(*, chat_id: int, assignments: dict[int, dict]) -> None:
    """Insert or replace the roles for several members in one batch.

    Args:
        chat_id: Group chat the roles belong to.
        assignments: Mapping of user_id to a dict with ``tag`` and ``reason``
            keys. An empty mapping is a no-op.
    """
    if not assignments:
        return
    now = time.time()
    rows = [
        (chat_id, user_id, entry["tag"], entry["reason"], now)
        for user_id, entry in assignments.items()
    ]
    async with database.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO user_tags (chat_id, user_id, tag, reason, assigned_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (chat_id, user_id) DO UPDATE
            SET tag = EXCLUDED.tag,
                reason = EXCLUDED.reason,
                assigned_at = EXCLUDED.assigned_at
            """,
            rows,
        )
