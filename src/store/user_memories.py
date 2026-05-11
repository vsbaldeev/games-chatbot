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

import re
import time

import numpy as np

from src.store import db as database

MAX_FACTS_PER_USER = 30


async def init_table() -> None:
    """Create the user_memories table, indexes, and embedding column if they do not exist."""
    async with database.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        async with conn.transaction():
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_memories (
                    id         BIGSERIAL        PRIMARY KEY,
                    chat_id    BIGINT           NOT NULL,
                    user_id    BIGINT           NOT NULL,
                    username   TEXT             NOT NULL,
                    fact       TEXT             NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_memories_lookup
                ON user_memories (chat_id, user_id)
            """)
            await conn.execute("""
                ALTER TABLE user_memories
                    ADD COLUMN IF NOT EXISTS embedding vector(384)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_memories_hnsw
                ON user_memories USING hnsw (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL
            """)


async def get_facts(*, chat_id: int, user_id: int) -> list[str]:
    """Return all facts for the user, newest first."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """SELECT fact FROM user_memories
               WHERE chat_id = $1 AND user_id = $2
               ORDER BY updated_at DESC""",
            chat_id, user_id,
        )
    return [row["fact"] for row in rows]


async def get_facts_for_users(
    *, chat_id: int, user_ids: list[int]
) -> dict[int, list[str]]:
    """Batch-fetch facts for several users, newest first per user."""
    if not user_ids:
        return {}
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """SELECT user_id, fact FROM user_memories
               WHERE chat_id = $1 AND user_id = ANY($2)
               ORDER BY updated_at DESC""",
            chat_id, user_ids,
        )
    result: dict[int, list[str]] = {}
    for row in rows:
        result.setdefault(row["user_id"], []).append(row["fact"])
    return result


async def get_facts_with_embeddings(*, chat_id: int, user_id: int) -> list[tuple[str, np.ndarray]]:
    """Return (fact, embedding) pairs for all facts that have embeddings stored, newest first."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """SELECT fact, embedding FROM user_memories
               WHERE chat_id = $1 AND user_id = $2 AND embedding IS NOT NULL
               ORDER BY updated_at DESC""",
            chat_id, user_id,
        )
    return [(row["fact"], np.array(row["embedding"])) for row in rows]


async def find_similar_fact(
    *, chat_id: int, user_id: int, embedding: list[float], threshold: float
) -> int | None:
    """Return the id of the most similar existing fact if similarity >= threshold, else None."""
    vector = np.array(embedding)
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, 1 - (embedding <=> $1) AS similarity
            FROM user_memories
            WHERE chat_id = $2 AND user_id = $3
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT 1
            """,
            vector, chat_id, user_id,
        )
    if row is None:
        return None
    return row["id"] if row["similarity"] >= threshold else None


async def refresh_updated_at(fact_id: int) -> None:
    """Bump updated_at on an existing fact to mark it as recently reinforced."""
    async with database.acquire() as conn:
        await conn.execute(
            "UPDATE user_memories SET updated_at = $1 WHERE id = $2",
            time.time(), fact_id,
        )


def format_hack_fact(count: int) -> str:
    last_two = count % 100
    last_one = count % 10
    if 11 <= last_two <= 19:
        suffix = "раз"
    elif last_one == 1:
        suffix = "раз"
    elif 2 <= last_one <= 4:
        suffix = "раза"
    else:
        suffix = "раз"
    return f"Пытался взломать бота {count} {suffix}"


async def upsert_hack_attempt(*, chat_id: int, user_id: int, username: str) -> None:
    """Increment (or create) the hack-attempt counter fact for this user."""
    now = time.time()
    async with database.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                SELECT id, fact FROM user_memories
                WHERE chat_id = $1 AND user_id = $2 AND fact LIKE 'Пытался взломать бота%'
                """,
                chat_id, user_id,
            )
            if rows:
                match = re.search(r"(\d+)", rows[0]["fact"])
                count = int(match.group(1)) + 1 if match else 2
                await conn.execute(
                    "UPDATE user_memories SET fact = $1, updated_at = $2 WHERE id = $3",
                    format_hack_fact(count), now, rows[0]["id"],
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO user_memories (chat_id, user_id, username, fact, updated_at)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    chat_id, user_id, username, format_hack_fact(1), now,
                )



async def upsert_facts(
    *, chat_id: int, user_id: int, username: str,
    facts: list[str], embeddings: list[list[float]]
) -> None:
    """Insert new facts with embeddings and prune oldest rows to stay within MAX_FACTS_PER_USER."""
    if not facts:
        return

    now = time.time()
    vectors = [np.array(emb) for emb in embeddings]
    async with database.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO user_memories (chat_id, user_id, username, fact, embedding, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                [(chat_id, user_id, username, fact, vector, now)
                 for fact, vector in zip(facts, vectors)],
            )
            await conn.execute(
                """
                DELETE FROM user_memories
                WHERE chat_id = $1 AND user_id = $2
                  AND id NOT IN (
                      SELECT id FROM user_memories
                      WHERE chat_id = $1 AND user_id = $2
                      ORDER BY updated_at DESC
                      LIMIT $3
                  )
                """,
                chat_id, user_id, MAX_FACTS_PER_USER,
            )
