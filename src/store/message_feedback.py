"""Per-message feedback facts: reactions, replies, corrections seen within
the first 15 minutes after a bot message is sent.

Every mutator's WHERE clause enforces the 15-minute window (except
add_correction — see its docstring), so a write arriving after the window
closed is a silent no-op rather than skewing a message's numbers with late
noise. Reads (used by the hourly rollup in feedback_metrics.py) are
unrestricted.
"""

from src.store import db as database

RETENTION_DAYS = 60
FEEDBACK_WINDOW_SQL = "interval '15 minutes'"


async def register(
    *, chat_id: int, message_id: int, source: str, trigger: str = "-", filter_verdict: str = "-",
) -> None:
    """Record a bot message as eligible for feedback tracking.

    Idempotent: ON CONFLICT DO NOTHING means a message registered twice
    (e.g. a search-notification message later edited to its final text)
    keeps its original sent_at from the first registration.

    Args:
        chat_id: Chat the message was sent in.
        message_id: The sent message's id.
        source: "pipeline" | "meme" | "selfie" | "roles" | "group_profile" | "notice".
        trigger: Pipeline response_trigger, or "-" for non-pipeline sources.
        filter_verdict: Filter node verdict, or "-" when not applicable.
    """
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO bot_message_feedback (chat_id, message_id, source, trigger, filter_verdict)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (chat_id, message_id) DO NOTHING
            """,
            chat_id, message_id, source, trigger, filter_verdict,
        )


async def add_reaction(*, chat_id: int, message_id: int, emoji: str) -> None:
    """Credit one added reaction, inside the 15-minute window only."""
    async with database.acquire() as conn:
        await conn.execute(
            f"""
            UPDATE bot_message_feedback
            SET reactions = reactions + 1,
                emojis = jsonb_set(emojis, ARRAY[$3], to_jsonb(COALESCE((emojis->>$3)::int, 0) + 1)),
                first_feedback_at = COALESCE(first_feedback_at, now())
            WHERE chat_id = $1 AND message_id = $2
              AND now() <= sent_at + {FEEDBACK_WINDOW_SQL}
            """,
            chat_id, message_id, emoji,
        )


async def remove_reaction(*, chat_id: int, message_id: int, emoji: str) -> None:
    """Credit one removed reaction, inside the 15-minute window only.

    Drops the emoji's key once its count reaches zero, rather than leaving
    a stale "0" entry in the JSONB map.
    """
    async with database.acquire() as conn:
        await conn.execute(
            f"""
            UPDATE bot_message_feedback
            SET reactions = GREATEST(reactions - 1, 0),
                emojis = CASE
                    WHEN COALESCE((emojis->>$3)::int, 0) <= 1 THEN emojis - $3
                    ELSE jsonb_set(emojis, ARRAY[$3], to_jsonb((emojis->>$3)::int - 1))
                END
            WHERE chat_id = $1 AND message_id = $2
              AND now() <= sent_at + {FEEDBACK_WINDOW_SQL}
            """,
            chat_id, message_id, emoji,
        )


async def add_reply(*, chat_id: int, message_id: int, depth: int) -> None:
    """Credit a reply landing in a bot message's thread, at hop depth >= 1.

    thread_depth tracks the deepest hop seen so far, not the latest.
    """
    async with database.acquire() as conn:
        await conn.execute(
            f"""
            UPDATE bot_message_feedback
            SET replies = replies + 1,
                thread_depth = GREATEST(thread_depth, $3),
                first_feedback_at = COALESCE(first_feedback_at, now())
            WHERE chat_id = $1 AND message_id = $2
              AND now() <= sent_at + {FEEDBACK_WINDOW_SQL}
            """,
            chat_id, message_id, depth,
        )


async def add_correction(*, chat_id: int, message_id: int) -> None:
    """Credit a classifier-confirmed correction.

    No window guard: the reply itself arrived inside the window (add_reply
    already required that), but the background classifier that labels it
    CORRECTION may finish after the window closes.
    """
    async with database.acquire() as conn:
        await conn.execute(
            "UPDATE bot_message_feedback SET corrections = corrections + 1 "
            "WHERE chat_id = $1 AND message_id = $2",
            chat_id, message_id,
        )


async def close_expired_windows() -> int:
    """Set `ignored` on every row whose 15-minute window has just closed.

    ignored = true when no reaction or reply arrived in time (first_feedback_at
    is still NULL); false otherwise. Rows already decided (ignored IS NOT NULL)
    are left alone.

    Returns:
        Number of rows updated.
    """
    async with database.acquire() as conn:
        result = await conn.execute(
            f"""
            UPDATE bot_message_feedback
            SET ignored = (first_feedback_at IS NULL)
            WHERE ignored IS NULL AND sent_at < now() - {FEEDBACK_WINDOW_SQL}
            """
        )
    return int(result.split()[-1])


async def cleanup_old() -> int:
    """Delete rows older than RETENTION_DAYS. Returns the number of deleted rows."""
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM bot_message_feedback WHERE sent_at < now() - make_interval(days => $1)",
            RETENTION_DAYS,
        )
    return int(result.split()[-1])
