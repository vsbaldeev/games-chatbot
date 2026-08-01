"""
Persistent store for the bot's own life canon — Жора's posted episodes and
the durable facts distilled from them.

Three row kinds share one table: ``episode`` rows are full posted life-story
entries, kept for narrative continuity when the next episode is written;
``fact`` rows are short durable canon sentences injected into chat replies;
``activity`` rows are silent daily-invented "what is Жора doing right now"
phrases (see ``src/life/activity.py``), generated without a chat post. Facts
survive episode/activity pruning, which is why all three kinds live together
rather than in separate tables — a completed state change («дом перекрашен»)
must outlive the episode that established it.

The newest of (episode, activity) rows' ``current_activity`` is the single
"what is Жора doing right now" answer — whichever was written most recently,
scheduled life post or daily refresh, wins.
"""

import time

import numpy as np

from src.store import db as database
from src.store import embedder

MAX_BOT_FACTS = 300
MAX_BOT_EPISODES = 100
MAX_BOT_ACTIVITIES = 30
BOT_FACT_SIMILARITY_THRESHOLD = 0.85

# current_activity decay windows, consumed by the context builder to bucket
# the newest activity into fresh / recent / stale phrasing. Owned here (not
# src/life/) because the context builder depends on this module regardless
# of whether the life-posting package exists yet. 14h (not 12h) so a 09:30
# MSK daily refresh stays "fresh" through the evening (23:30) instead of
# flipping to "recent" at 21:30 while the chat is still awake.
ACTIVITY_FRESH_HOURS = 14
ACTIVITY_RECENT_HOURS = 48

EPISODE_SIMILARITY_FLOOR = 0.55


async def get_recent_episodes(limit: int) -> list[dict]:
    """Return the most recent posted episodes, newest first.

    Args:
        limit: Maximum number of episodes to return.

    Returns:
        Rows with ``content``, ``post_format``, ``current_activity`` and
        ``posted_at``, newest first.
    """
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content, post_format, current_activity, posted_at
            FROM bot_memories
            WHERE kind = 'episode'
            ORDER BY posted_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [dict(row) for row in rows]


async def get_latest_posted_at() -> float | None:
    """Return the ``posted_at`` of the most recent episode, or None if none exist."""
    async with database.acquire() as conn:
        value = await conn.fetchval(
            "SELECT MAX(posted_at) FROM bot_memories WHERE kind = 'episode'"
        )
    return value


async def get_current_activity() -> tuple[str, float] | None:
    """Return the newest current-activity phrase and its timestamp.

    Considers both ``episode`` rows (set by a scheduled life post) and
    ``activity`` rows (set by the silent daily refresh) — whichever is
    newest wins, so a life post always supersedes a same-day refresh and
    vice versa.

    Returns:
        ``(phrase, posted_at)`` for the newest row that carries a
        current-activity phrase, or None when none exists yet.
    """
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT current_activity, posted_at
            FROM bot_memories
            WHERE kind IN ('episode', 'activity') AND current_activity IS NOT NULL
            ORDER BY posted_at DESC
            LIMIT 1
            """
        )
    if row is None:
        return None
    return row["current_activity"], row["posted_at"]


async def get_recent_activities(limit: int) -> list[tuple[str, float]]:
    """Return recent current-activity phrases with their timestamps, newest first.

    Spans both ``episode`` and ``activity`` rows, so callers can render a
    dated history of what Жора has been doing regardless of whether each
    entry came from a scheduled life post or a silent daily refresh.

    Args:
        limit: Maximum number of entries to return.

    Returns:
        ``(phrase, posted_at)`` pairs, newest first.
    """
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT current_activity, posted_at
            FROM bot_memories
            WHERE kind IN ('episode', 'activity') AND current_activity IS NOT NULL
            ORDER BY posted_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [(row["current_activity"], row["posted_at"]) for row in rows]


async def insert_activity(phrase: str) -> None:
    """Insert a silently-generated daily activity phrase and prune old ones.

    Args:
        phrase: Present-tense activity phrase, already validated for length.
    """
    now = time.time()
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO bot_memories
                (kind, content, current_activity, posted_at, created_at, updated_at)
            VALUES ('activity', $1, $1, $2, $2, $2)
            """,
            phrase, now,
        )
    await prune_activities(keep=MAX_BOT_ACTIVITIES)


async def prune_activities(keep: int = MAX_BOT_ACTIVITIES) -> None:
    """Delete activity rows beyond the newest ``keep``, oldest first.

    Args:
        keep: Number of newest activity rows to retain.
    """
    async with database.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM bot_memories
            WHERE kind = 'activity'
              AND id NOT IN (
                  SELECT id FROM bot_memories
                  WHERE kind = 'activity'
                  ORDER BY posted_at DESC
                  LIMIT $1
              )
            """,
            keep,
        )


async def insert_episode(
    *,
    content: str,
    post_format: str,
    current_activity: str | None,
    embedding: list[float],
) -> None:
    """Insert a newly posted episode and prune old ones beyond the cap.

    Args:
        content: Full episode text (the caption/story the chat saw).
        post_format: Format the episode was posted as (``story``, ``voice``…).
        current_activity: Present-tense activity phrase, or None.
        embedding: 384-dim embedding of ``content``.
    """
    now = time.time()
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO bot_memories
                (kind, content, embedding, post_format, posted_at,
                 current_activity, created_at, updated_at)
            VALUES ('episode', $1, $2, $3, $4, $5, $4, $4)
            """,
            content, np.array(embedding), post_format, now, current_activity,
        )
    await prune_episodes(keep=MAX_BOT_EPISODES)


async def prune_episodes(keep: int = MAX_BOT_EPISODES) -> None:
    """Delete episode rows beyond the newest ``keep``, oldest first.

    Args:
        keep: Number of newest episodes to retain.
    """
    async with database.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM bot_memories
            WHERE kind = 'episode'
              AND id NOT IN (
                  SELECT id FROM bot_memories
                  WHERE kind = 'episode'
                  ORDER BY posted_at DESC
                  LIMIT $1
              )
            """,
            keep,
        )


async def get_facts(limit: int) -> list[str]:
    """Return up to ``limit`` canon facts, newest first."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content FROM bot_memories
            WHERE kind = 'fact'
            ORDER BY updated_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [row["content"] for row in rows]


async def get_writer_facts(*, newest: int = 20, sampled: int = 10) -> list[str]:
    """Return canon facts for the episode writer: recent state plus long-tail callbacks.

    Args:
        newest: Number of most-recently-updated facts to include in full.
        sampled: Number of additional older facts to sample at random, so
            the writer can occasionally callback to long-forgotten canon
            without ever reading the entire fact store.

    Returns:
        Newest facts first, followed by the random sample.
    """
    async with database.acquire() as conn:
        newest_rows = await conn.fetch(
            """
            SELECT id, content FROM bot_memories
            WHERE kind = 'fact'
            ORDER BY updated_at DESC
            LIMIT $1
            """,
            newest,
        )
        newest_ids = [row["id"] for row in newest_rows]
        sampled_rows = await conn.fetch(
            """
            SELECT content FROM bot_memories
            WHERE kind = 'fact' AND id != ALL($1::bigint[])
            ORDER BY random()
            LIMIT $2
            """,
            newest_ids, sampled,
        )
    return [row["content"] for row in newest_rows] + [row["content"] for row in sampled_rows]


async def find_similar_facts(embedding: list[float], top_k: int) -> list[str]:
    """Return canon facts most similar to ``embedding``, closest first.

    Args:
        embedding: Query embedding (e.g. of the incoming chat message).
        top_k: Maximum number of facts to return.
    """
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content FROM bot_memories
            WHERE kind = 'fact' AND embedding IS NOT NULL
            ORDER BY embedding <=> $1, updated_at DESC
            LIMIT $2
            """,
            np.array(embedding), top_k,
        )
    return [row["content"] for row in rows]


async def find_similar_episodes(
    embedding: list[float], top_k: int = 2, floor: float = EPISODE_SIMILARITY_FLOOR
) -> list[str]:
    """Return past episodes similar to ``embedding``, above a similarity floor.

    Args:
        embedding: Query embedding (e.g. of the incoming chat message).
        top_k: Maximum number of episodes to return.
        floor: Minimum cosine similarity required to include an episode —
            keeps unrelated full episode texts out of the prompt.
    """
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content, 1 - (embedding <=> $1) AS similarity
            FROM bot_memories
            WHERE kind = 'episode' AND embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            np.array(embedding), top_k,
        )
    return [row["content"] for row in rows if row["similarity"] >= floor]


async def find_similar_bot_fact(embedding: list[float], threshold: float) -> int | None:
    """Return the id of the most similar existing fact if similarity >= threshold.

    Mirrors ``user_memories.find_similar_fact`` for the bot's own fact rows.
    """
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, 1 - (embedding <=> $1) AS similarity
            FROM bot_memories
            WHERE kind = 'fact' AND embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT 1
            """,
            np.array(embedding),
        )
    if row is None:
        return None
    return row["id"] if row["similarity"] >= threshold else None


async def upsert_facts(facts: list[str]) -> None:
    """Insert or refresh canon facts, newest text winning on a semantic match.

    Each fact is embedded and matched against existing facts by cosine
    similarity. A match replaces the stored text and embedding (state
    changes like «дом перекрашен» overwrite the old state instead of
    accumulating); no match inserts a new row. Facts beyond ``MAX_BOT_FACTS``
    are pruned, oldest by ``updated_at`` first.

    Args:
        facts: New fact sentences distilled from a posted episode.
    """
    if not facts:
        return
    now = time.time()
    for fact in facts:
        embedding = await embedder.embed(fact)
        matched_id = await find_similar_bot_fact(embedding, BOT_FACT_SIMILARITY_THRESHOLD)
        if matched_id is not None:
            await replace_fact(matched_id, fact, embedding, now)
        else:
            await insert_fact(fact, embedding, now)
    await prune_facts(keep=MAX_BOT_FACTS)


async def replace_fact(fact_id: int, content: str, embedding: list[float], now: float) -> None:
    """Overwrite an existing fact row's text, embedding and updated_at."""
    async with database.acquire() as conn:
        await conn.execute(
            "UPDATE bot_memories SET content = $1, embedding = $2, updated_at = $3 WHERE id = $4",
            content, np.array(embedding), now, fact_id,
        )


async def insert_fact(content: str, embedding: list[float], now: float) -> None:
    """Insert a brand-new fact row."""
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO bot_memories (kind, content, embedding, created_at, updated_at)
            VALUES ('fact', $1, $2, $3, $3)
            """,
            content, np.array(embedding), now,
        )


async def prune_facts(keep: int = MAX_BOT_FACTS) -> None:
    """Delete fact rows beyond the newest ``keep`` (by updated_at)."""
    async with database.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM bot_memories
            WHERE kind = 'fact'
              AND id NOT IN (
                  SELECT id FROM bot_memories
                  WHERE kind = 'fact'
                  ORDER BY updated_at DESC
                  LIMIT $1
              )
            """,
            keep,
        )
