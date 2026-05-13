"""
Persistent store for roast event log and emoji reaction tracking.

Captures which anchor type was used per roast and how users reacted,
enabling future anchor-selection weighting based on engagement signals.
"""

import random
import time

from src.store import db as database


async def init_tables() -> None:
    async with database.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS roast_log (
                    message_id     BIGINT           NOT NULL,
                    chat_id        BIGINT           NOT NULL,
                    target_user_id BIGINT           NOT NULL,
                    anchor_key     TEXT             NOT NULL,
                    created_at     DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (message_id, chat_id)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS roast_reactions (
                    message_id  BIGINT  NOT NULL,
                    chat_id     BIGINT  NOT NULL,
                    emoji       TEXT    NOT NULL,
                    count       INT     NOT NULL DEFAULT 0,
                    PRIMARY KEY (message_id, chat_id, emoji)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS roast_queue (
                    chat_id BIGINT NOT NULL,
                    user_id BIGINT NOT NULL,
                    PRIMARY KEY (chat_id, user_id)
                )
            """)


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


async def is_roast_message(message_id: int, chat_id: int) -> bool:
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM roast_log WHERE message_id = $1 AND chat_id = $2",
            message_id, chat_id,
        )
    return row is not None


async def record_reaction(message_id: int, chat_id: int, emoji: str, delta: int) -> None:
    """Upsert a reaction count delta; count floor is zero."""
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO roast_reactions (message_id, chat_id, emoji, count)
            VALUES ($1, $2, $3, GREATEST(0, $4))
            ON CONFLICT (message_id, chat_id, emoji)
            DO UPDATE SET count = GREATEST(0, roast_reactions.count + $4)
            """,
            message_id, chat_id, emoji, delta,
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


async def get_anchor_stats(chat_id: int) -> dict[str, int]:
    """Return total reaction counts per anchor key for a chat."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT rl.anchor_key, COALESCE(SUM(rr.count), 0) AS total
            FROM roast_log rl
            LEFT JOIN roast_reactions rr
                ON rr.message_id = rl.message_id AND rr.chat_id = rl.chat_id
            WHERE rl.chat_id = $1
            GROUP BY rl.anchor_key
            """,
            chat_id,
        )
    return {row["anchor_key"]: int(row["total"]) for row in rows}
