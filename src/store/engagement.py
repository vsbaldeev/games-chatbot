"""
Persistent store for per-user attention scores (the conversation wind-down
engine's leaky bucket).

One row per (chat_id, user_id). The score decays exponentially with a fixed
half-life; decay is computed lazily inside SQL on every access, so the row
only stores the score and the moment it was last valid at. Writing is a
single atomic ``INSERT … ON CONFLICT DO UPDATE … RETURNING`` — Postgres
row locking serializes concurrent messages from the same user, so there is
no read-modify-write race in Python.
"""

from src.store import db as database


async def add_signal(
    *, chat_id: int, user_id: int, weight: float,
    now: float, half_life_seconds: float,
) -> float:
    """Atomically decay the stored score to ``now``, add ``weight``, return it.

    ``GREATEST(elapsed, 0)`` caps the decay factor at 1 so wall-clock skew can
    never inflate the score, and ``last_signal_at`` stays monotone.

    Args:
        chat_id: Chat the score belongs to.
        user_id: User the score is tracked for.
        weight: Signal weight to add after decaying.
        now: Current unix timestamp.
        half_life_seconds: Seconds for the score to halve while quiet.

    Returns:
        The new decayed-plus-charged score.
    """
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO engagement_scores (chat_id, user_id, score, last_signal_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (chat_id, user_id) DO UPDATE
            SET score = engagement_scores.score
                        * POWER(0.5, GREATEST($4 - engagement_scores.last_signal_at, 0) / $5)
                        + $3,
                last_signal_at = GREATEST(engagement_scores.last_signal_at, $4)
            RETURNING score
            """,
            chat_id, user_id, weight, now, half_life_seconds,
        )
    return float(row["score"])


async def peek_score(
    *, chat_id: int, user_id: int,
    now: float, half_life_seconds: float,
) -> float:
    """Return the decayed score without charging it (read-only).

    Used by consumers that must not spend the user's budget, e.g. the humor
    node checking whether a joke target is wound down.

    Args:
        chat_id: Chat the score belongs to.
        user_id: User the score is tracked for.
        now: Current unix timestamp.
        half_life_seconds: Seconds for the score to halve while quiet.

    Returns:
        The decayed score, or 0.0 when the user has no row yet.
    """
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT score * POWER(0.5, GREATEST($3 - last_signal_at, 0) / $4) AS score
            FROM engagement_scores
            WHERE chat_id = $1 AND user_id = $2
            """,
            chat_id, user_id, now, half_life_seconds,
        )
    return float(row["score"]) if row else 0.0
