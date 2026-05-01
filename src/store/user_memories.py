"""
Persistent store for per-user facts extracted by the MemoryWriter after each
bot response.

Facts are plain-language sentences produced by a lightweight LLM, for example:
  "Prefers co-op games over singleplayer"
  "Asked about Elden Ring twice in October"
  "Has won 5 duels this month"

The table is keyed by (chat_id, user_id) so the same user is tracked separately
in different group chats.
"""

import time

from src.store import db as database

# How many facts to keep per user.  Oldest rows are pruned when the limit is hit.
MAX_FACTS_PER_USER = 10


async def init_table() -> None:
    """Create the user_memories table and its lookup index if they do not exist."""
    db = await database.get()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS user_memories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            username    TEXT    NOT NULL,
            fact        TEXT    NOT NULL,
            updated_at  REAL    NOT NULL
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_memories_lookup
        ON user_memories (chat_id, user_id)
    """)
    await db.commit()


async def get_facts(*, chat_id: int, user_id: int) -> list[str]:
    """Return all facts stored for the given user in this chat, newest first."""
    db = await database.get()
    rows = await db.execute_fetchall(
        """
        SELECT fact FROM user_memories
        WHERE chat_id = ? AND user_id = ?
        ORDER BY updated_at DESC
        """,
        (chat_id, user_id),
    )
    return [row["fact"] for row in rows]


async def get_facts_for_users(
    *, chat_id: int, user_ids: list[int]
) -> dict[int, list[str]]:
    """
    Batch-fetch facts for several users in a single DB round-trip.

    Returns a mapping of user_id → list[fact].  Users with no stored facts
    are not included in the result.
    """
    if not user_ids:
        return {}

    placeholders = ", ".join("?" * len(user_ids))
    db = await database.get()
    rows = await db.execute_fetchall(
        f"""
        SELECT user_id, fact FROM user_memories
        WHERE chat_id = ? AND user_id IN ({placeholders})
        ORDER BY updated_at DESC
        """,
        (chat_id, *user_ids),
    )

    result: dict[int, list[str]] = {}
    for row in rows:
        result.setdefault(row["user_id"], []).append(row["fact"])
    return result


async def upsert_facts(
    *, chat_id: int, user_id: int, username: str, facts: list[str]
) -> None:
    """
    Insert new facts for a user and prune the oldest rows so the total stays
    within MAX_FACTS_PER_USER.
    """
    if not facts:
        return

    now = time.time()
    db = await database.get()
    for fact in facts:
        await db.execute(
            """
            INSERT INTO user_memories (chat_id, user_id, username, fact, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_id, user_id, username, fact, now),
        )

    # Prune oldest facts beyond the cap.
    await db.execute(
        """
        DELETE FROM user_memories
        WHERE chat_id = ? AND user_id = ?
          AND id NOT IN (
              SELECT id FROM user_memories
              WHERE chat_id = ? AND user_id = ?
              ORDER BY updated_at DESC
              LIMIT ?
          )
        """,
        (chat_id, user_id, chat_id, user_id, MAX_FACTS_PER_USER),
    )
    await db.commit()
