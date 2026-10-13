"""Per-message feedback counters for every bot message.

One row per bot message, written by register() at send time. Reactions and
replies move counters whenever they arrive, for as long as the row exists —
there is no time window. `ignored` is kept equal to "every counter is 0" by
every mutating statement, so removing the last reaction makes a message
ignored again.

Columns are always chosen from COUNTER_COLUMNS, never interpolated from input.
"""

from src.feedback.sentiment import is_positive_reaction
from src.store import db as database

RETENTION_DAYS = 60

COUNTER_COLUMNS: tuple[str, ...] = (
    "positive_reactions",
    "negative_reactions",
    "positive_replies",
    "negative_replies",
    "neutral_replies",
    "media_replies",
    "unlabelled_replies",
)

REPLY_KIND_COLUMNS: dict[str, str] = {
    "positive": "positive_replies",
    "negative": "negative_replies",
    "neutral": "neutral_replies",
    "media": "media_replies",
    "unlabelled": "unlabelled_replies",
}


def reaction_column(emoji: str) -> str:
    """Pick the reaction counter for one emoji key."""
    return "positive_reactions" if is_positive_reaction(emoji) else "negative_reactions"


def require_counter(column: str) -> None:
    """Refuse any column that is not a known counter."""
    if column not in COUNTER_COLUMNS:
        raise ValueError(f"not a feedback counter: {column!r}")


def increment_sql(column: str) -> str:
    """UPDATE that adds one to `column`; any increment means the message got a response."""
    require_counter(column)
    return f"""
        UPDATE bot_message_feedback
        SET {column} = {column} + 1, ignored = FALSE
        WHERE chat_id = $1 AND message_id = $2
    """


def decrement_sql(column: str) -> str:
    """UPDATE that subtracts one from `column`, floored at 0, and recomputes `ignored`.

    SET expressions see the pre-update row, so `ignored` is computed from the
    decremented value of `column` plus the unchanged others.
    """
    require_counter(column)
    decremented = f"GREATEST({column} - 1, 0)"
    others = " + ".join(other for other in COUNTER_COLUMNS if other != column)
    return f"""
        UPDATE bot_message_feedback
        SET {column} = {decremented}, ignored = ({decremented} + {others} = 0)
        WHERE chat_id = $1 AND message_id = $2
    """


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
    """Count one added reaction. A no-op for a message with no feedback row."""
    async with database.acquire() as conn:
        await conn.execute(increment_sql(reaction_column(emoji)), chat_id, message_id)


async def remove_reaction(*, chat_id: int, message_id: int, emoji: str) -> None:
    """Uncount one removed reaction, never below zero."""
    async with database.acquire() as conn:
        await conn.execute(decrement_sql(reaction_column(emoji)), chat_id, message_id)


async def add_reply(*, chat_id: int, message_id: int, kind: str) -> None:
    """Count one reply of `kind` (a key of REPLY_KIND_COLUMNS) on a bot message."""
    column = REPLY_KIND_COLUMNS[kind]
    async with database.acquire() as conn:
        await conn.execute(increment_sql(column), chat_id, message_id)


async def get_origin(*, chat_id: int, message_id: int) -> tuple[str, str] | None:
    """Return (source, trigger) for a tracked bot message, or None if it has no row."""
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT source, trigger FROM bot_message_feedback WHERE chat_id = $1 AND message_id = $2",
            chat_id, message_id,
        )
    return None if row is None else (row["source"], row["trigger"])


async def cleanup_old() -> int:
    """Delete rows older than RETENTION_DAYS. Returns the number of deleted rows."""
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM bot_message_feedback WHERE sent_at < now() - make_interval(days => $1)",
            RETENTION_DAYS,
        )
    return int(result.split()[-1])
