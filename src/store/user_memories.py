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

# Prefixes of counter-tally facts (see format_insult_fact / format_hack_fact).
# These are bookkeeping for weekly roles and roasts, not conversational
# memory: surfacing them in ordinary replies reads as the bot holding a
# grudge, so reply-context assembly filters them out via is_counter_fact.
COUNTER_FACT_PREFIXES = ("Оскорблял бота", "Пытался взломать бота")

# Facts untouched for this long are stale residue: real, current facts get
# their updated_at refreshed by the dedup path whenever they are re-observed.
# Kept short so jokes/roasts about members stay current rather than dredging
# up things from months ago.
FACT_RETENTION_DAYS = 14


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


def pluralize_times(count: int) -> str:
    """Return the Russian plural form of «раз» for the given count.

    Args:
        count: Number the word agrees with.

    Returns:
        ``"раз"`` or ``"раза"`` depending on Russian pluralization rules.
    """
    last_two = count % 100
    last_one = count % 10
    if 11 <= last_two <= 19:
        return "раз"
    if last_one == 1:
        return "раз"
    if 2 <= last_one <= 4:
        return "раза"
    return "раз"


def is_counter_fact(fact: str) -> bool:
    """Tell whether a stored fact is a counter tally (insults, hack attempts).

    Args:
        fact: A stored ``user_memories`` fact string.

    Returns:
        True when the fact is one of the counter facts listed in
        ``COUNTER_FACT_PREFIXES``.
    """
    return fact.startswith(COUNTER_FACT_PREFIXES)


def format_hack_fact(count: int) -> str:
    """Format the hack-attempt counter fact text.

    Args:
        count: Total number of recorded hack attempts.

    Returns:
        Fact string, e.g. ``"Пытался взломать бота 3 раза"``.
    """
    return f"Пытался взломать бота {count} {pluralize_times(count)}"


def format_insult_fact(count: int) -> str:
    """Format the bot-insult counter fact text.

    Args:
        count: Total number of recorded insults aimed at the bot.

    Returns:
        Fact string, e.g. ``"Оскорблял бота 5 раз"``.
    """
    return f"Оскорблял бота {count} {pluralize_times(count)}"


async def upsert_counter_fact(
    *, chat_id: int, user_id: int, username: str,
    like_pattern: str, format_fact,
) -> None:
    """Increment (or create) a counter-style fact matched by ``like_pattern``.

    Args:
        chat_id: Chat the fact belongs to.
        user_id: User the fact is about.
        username: Current username of the user.
        like_pattern: SQL LIKE pattern identifying the counter fact row.
        format_fact: Callable mapping a count to the fact string.
    """
    now = time.time()
    async with database.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                SELECT id, fact FROM user_memories
                WHERE chat_id = $1 AND user_id = $2 AND fact LIKE $3
                """,
                chat_id, user_id, like_pattern,
            )
            if rows:
                match = re.search(r"(\d+)", rows[0]["fact"])
                count = int(match.group(1)) + 1 if match else 2
                await conn.execute(
                    "UPDATE user_memories SET fact = $1, updated_at = $2 WHERE id = $3",
                    format_fact(count), now, rows[0]["id"],
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO user_memories (chat_id, user_id, username, fact, updated_at)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    chat_id, user_id, username, format_fact(1), now,
                )


async def upsert_hack_attempt(*, chat_id: int, user_id: int, username: str) -> None:
    """Increment (or create) the hack-attempt counter fact for this user."""
    await upsert_counter_fact(
        chat_id=chat_id, user_id=user_id, username=username,
        like_pattern="Пытался взломать бота%", format_fact=format_hack_fact,
    )


async def upsert_insult_attempt(*, chat_id: int, user_id: int, username: str) -> None:
    """Increment (or create) the bot-insult counter fact for this user."""
    await upsert_counter_fact(
        chat_id=chat_id, user_id=user_id, username=username,
        like_pattern="Оскорблял бота%", format_fact=format_insult_fact,
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


async def cleanup_stale(*, days: int = FACT_RETENTION_DAYS) -> int:
    """Delete facts whose ``updated_at`` is older than ``days`` days.

    Applies to every fact, counter facts (insults, hack attempts) included —
    a counter untouched for the whole window is stale by the same standard.
    Genuinely repeated facts survive because the dedup path refreshes
    ``updated_at`` on every re-observation.

    Args:
        days: Retention window in days.

    Returns:
        Number of deleted rows.
    """
    cutoff = time.time() - days * 86400
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM user_memories WHERE updated_at < $1",
            cutoff,
        )
    return int(result.split()[-1])
