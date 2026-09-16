"""
Persistent store for the roast event log.

Captures which anchor type was used per roast, and the round-robin
target queue.
"""

import random
import time

from src.store import db as database


async def log_roast(
    *,
    message_id: int,
    chat_id: int,
    target_user_id: int,
    anchor_key: str,
) -> None:
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO roast_log (message_id, chat_id, target_user_id, anchor_key, created_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
            """,
            message_id, chat_id, target_user_id, anchor_key, time.time(),
        )


async def pop_roast_target(chat_id: int, members: list[tuple[int, str]]) -> tuple[int, str]:
    """Pick the next roast target using a shuffle-bag cycle.

    Draws randomly from members not yet roasted this cycle. When the cycle is
    exhausted, refills the queue with all current members and starts a new cycle.
    """
    member_map = {uid: uname for uid, uname in members}
    async with database.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                "SELECT user_id FROM roast_queue WHERE chat_id = $1 AND user_id = ANY($2)",
                chat_id, list(member_map),
            )
            remaining = {row["user_id"] for row in rows}
            if not remaining:
                await conn.execute("DELETE FROM roast_queue WHERE chat_id = $1", chat_id)
                await conn.executemany(
                    "INSERT INTO roast_queue (chat_id, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    [(chat_id, uid) for uid in member_map],
                )
                remaining = set(member_map)
            target_id = random.choice(list(remaining))
            await conn.execute(
                "DELETE FROM roast_queue WHERE chat_id = $1 AND user_id = $2",
                chat_id, target_id,
            )
    return target_id, member_map[target_id]


async def get_recent_modes(chat_id: int, user_id: int, limit: int) -> list[str]:
    """Return the anchor keys of this user's most recent roasts, newest first.

    Lets the roast picker avoid repeating the angle used last time. Reads the
    existing ``roast_log`` table, so no extra state is stored.

    Args:
        chat_id: Telegram chat ID the roasts belong to.
        user_id: Telegram user ID of the roast target.
        limit: Maximum number of recent anchor keys to return.

    Returns:
        Up to ``limit`` anchor keys ordered from most to least recent.
    """
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT anchor_key FROM roast_log
            WHERE chat_id = $1 AND target_user_id = $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            chat_id, user_id, limit,
        )
    return [row["anchor_key"] for row in rows]
