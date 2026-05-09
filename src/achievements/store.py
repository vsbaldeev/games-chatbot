"""
Async database operations for the achievement system.
"""

import time

from src.achievements.definitions import TRACKABLE_STATS, MAX_TRACKABLE_STATS
from src.store import db as database


async def create_core_tables(conn) -> None:
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_members (
            chat_id  BIGINT NOT NULL,
            user_id  BIGINT NOT NULL,
            username TEXT,
            PRIMARY KEY (chat_id, user_id)
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id              BIGINT  NOT NULL,
            chat_id              BIGINT  NOT NULL,
            username             TEXT,
            laugh_reactions      INTEGER NOT NULL DEFAULT 0,
            heart_reactions      INTEGER NOT NULL DEFAULT 0,
            fire_reactions       INTEGER NOT NULL DEFAULT 0,
            thumbsup_reactions   INTEGER NOT NULL DEFAULT 0,
            emoji_messages       INTEGER NOT NULL DEFAULT 0,
            sticker_messages     INTEGER NOT NULL DEFAULT 0,
            forwarded_messages   INTEGER NOT NULL DEFAULT 0,
            link_messages        INTEGER NOT NULL DEFAULT 0,
            voice_messages       INTEGER NOT NULL DEFAULT 0,
            video_messages       INTEGER NOT NULL DEFAULT 0,
            video_note_messages  INTEGER NOT NULL DEFAULT 0,
            photo_messages       INTEGER NOT NULL DEFAULT 0,
            night_messages       INTEGER NOT NULL DEFAULT 0,
            animation_messages   INTEGER NOT NULL DEFAULT 0,
            roasted_count        INTEGER NOT NULL DEFAULT 0,
            roulette_win_count   INTEGER NOT NULL DEFAULT 0,
            duel_wins            INTEGER NOT NULL DEFAULT 0,
            long_messages        INTEGER NOT NULL DEFAULT 0,
            voice_max_duration   INTEGER NOT NULL DEFAULT 0,
            long_message_max     INTEGER NOT NULL DEFAULT 0,
            last_seen            BIGINT  NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, chat_id)
        )
    """)


async def create_event_tables(conn) -> None:
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS announced_achievements (
            user_id BIGINT NOT NULL,
            chat_id BIGINT NOT NULL,
            key     TEXT   NOT NULL,
            PRIMARY KEY (user_id, chat_id, key)
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS message_authors (
            chat_id    BIGINT NOT NULL,
            message_id BIGINT NOT NULL,
            user_id    BIGINT NOT NULL,
            username   TEXT   NOT NULL,
            created_at BIGINT NOT NULL,
            PRIMARY KEY (chat_id, message_id)
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS message_reaction_counts (
            chat_id     BIGINT  NOT NULL,
            message_id  BIGINT  NOT NULL,
            emoji       TEXT    NOT NULL,
            total_count INTEGER NOT NULL,
            updated_at  BIGINT  NOT NULL,
            PRIMARY KEY (chat_id, message_id, emoji)
        )
    """)


async def init_tables() -> None:
    """Create all achievement-related tables."""
    async with database.acquire() as conn:
        async with conn.transaction():
            await create_core_tables(conn)
            await create_event_tables(conn)


async def register_member(chat_id: int, user_id: int, username: str) -> None:
    """Add a user to chat_members if not already present (no-op on conflict)."""
    async with database.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_members (chat_id, user_id, username) "
            "VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            chat_id, user_id, username,
        )


async def get_chat_members(chat_id: int) -> list[tuple[int, str]]:
    """Return (user_id, username) pairs for every member registered in this chat."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id, username FROM chat_members WHERE chat_id = $1",
            chat_id,
        )
    return [(row["user_id"], row["username"] or f"user_{row['user_id']}") for row in rows]


async def get_all_chat_ids() -> list[int]:
    """Return all distinct chat IDs that have at least one registered member."""
    async with database.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT chat_id FROM chat_members")
    return [row["chat_id"] for row in rows]


async def increment_stat(user_id: int, chat_id: int, username: str, stat: str) -> int:
    """Increment a counter stat by 1 and return the new value."""
    if stat not in TRACKABLE_STATS:
        raise ValueError(f"Unknown stat '{stat}'. Allowed: {TRACKABLE_STATS}")
    now = int(time.time())
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            f"""INSERT INTO user_stats (user_id, chat_id, username, {stat}, last_seen)
                VALUES ($1, $2, $3, 1, $4)
                ON CONFLICT (user_id, chat_id) DO UPDATE SET
                    {stat}    = user_stats.{stat} + 1,
                    username  = EXCLUDED.username,
                    last_seen = EXCLUDED.last_seen
                RETURNING {stat}""",
            user_id, chat_id, username, now,
        )
        return row[stat]


async def update_max_stat(user_id: int, chat_id: int, username: str, stat: str, value: int) -> None:
    """Update a max-tracking stat if value exceeds the stored maximum."""
    if stat not in MAX_TRACKABLE_STATS:
        raise ValueError(f"Unknown max stat '{stat}'. Allowed: {MAX_TRACKABLE_STATS}")
    now = int(time.time())
    async with database.acquire() as conn:
        await conn.execute(
            f"""INSERT INTO user_stats (user_id, chat_id, username, {stat}, last_seen)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, chat_id) DO UPDATE SET
                    {stat}    = GREATEST(user_stats.{stat}, EXCLUDED.{stat}),
                    username  = EXCLUDED.username,
                    last_seen = EXCLUDED.last_seen""",
            user_id, chat_id, username, value, now,
        )


async def get_user_stats(user_id: int, chat_id: int) -> dict[str, int]:
    """Return the full stat row for a user in a chat, or {} if no row exists."""
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT laugh_reactions, heart_reactions, fire_reactions, thumbsup_reactions,
                      emoji_messages, sticker_messages, forwarded_messages,
                      link_messages, voice_messages, video_messages, video_note_messages,
                      photo_messages, night_messages, long_messages,
                      voice_max_duration, long_message_max,
                      roasted_count, duel_wins, animation_messages
               FROM user_stats WHERE user_id = $1 AND chat_id = $2""",
            user_id, chat_id,
        )
    if not row:
        return {}
    return {
        "laugh_reactions":     row["laugh_reactions"],
        "heart_reactions":     row["heart_reactions"],
        "fire_reactions":      row["fire_reactions"],
        "thumbsup_reactions":  row["thumbsup_reactions"],
        "emoji_messages":      row["emoji_messages"],
        "sticker_messages":    row["sticker_messages"],
        "forwarded_messages":  row["forwarded_messages"],
        "link_messages":       row["link_messages"],
        "voice_messages":      row["voice_messages"],
        "video_messages":      row["video_messages"],
        "video_note_messages": row["video_note_messages"],
        "photo_messages":      row["photo_messages"],
        "night_messages":      row["night_messages"],
        "long_messages":       row["long_messages"],
        "voice_max_duration":  row["voice_max_duration"],
        "long_message_max":    row["long_message_max"],
        "roasted_count":       row["roasted_count"],
        "duel_wins":           row["duel_wins"],
        "animation_messages":  row["animation_messages"],
    }


async def get_announced_keys(user_id: int, chat_id: int) -> set[str]:
    """Return all achievement keys already announced to this user in this chat."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            "SELECT key FROM announced_achievements WHERE user_id = $1 AND chat_id = $2",
            user_id, chat_id,
        )
    return {row["key"] for row in rows}


async def mark_and_get_new(user_id: int, chat_id: int, keys: list[str]) -> list[str]:
    """Insert keys into announced_achievements atomically; return only newly inserted keys."""
    if not keys:
        return []
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """INSERT INTO announced_achievements (user_id, chat_id, key)
               SELECT $1, $2, unnest($3::text[])
               ON CONFLICT DO NOTHING
               RETURNING key""",
            user_id, chat_id, keys,
        )
    return [row["key"] for row in rows]


async def set_message_author(
    chat_id: int, message_id: int, user_id: int, username: str
) -> None:
    """Record which user authored a message (for reaction credit lookups)."""
    async with database.acquire() as conn:
        await conn.execute(
            "INSERT INTO message_authors "
            "(chat_id, message_id, user_id, username, created_at) "
            "VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING",
            chat_id, message_id, user_id, username, int(time.time()),
        )


async def get_message_author(chat_id: int, message_id: int) -> tuple[int, str] | None:
    """Return (user_id, username) for the author of a message, or None if unknown."""
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id, username FROM message_authors "
            "WHERE chat_id = $1 AND message_id = $2",
            chat_id, message_id,
        )
    return (row["user_id"], row["username"]) if row else None


async def apply_reaction_counts(
    chat_id: int, message_id: int, new_counts: dict[str, int]
) -> dict[str, int]:
    """Persist the latest per-emoji reaction totals and return positive deltas per emoji."""
    deltas: dict[str, int] = {}
    now = int(time.time())
    async with database.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                "SELECT emoji, total_count FROM message_reaction_counts "
                "WHERE chat_id = $1 AND message_id = $2",
                chat_id, message_id,
            )
            previous = {row["emoji"]: row["total_count"] for row in rows}
            for emoji, new_total in new_counts.items():
                delta = max(0, new_total - previous.get(emoji, 0))
                if delta > 0:
                    deltas[emoji] = delta
                await conn.execute(
                    """INSERT INTO message_reaction_counts
                           (chat_id, message_id, emoji, total_count, updated_at)
                       VALUES ($1, $2, $3, $4, $5)
                       ON CONFLICT (chat_id, message_id, emoji) DO UPDATE SET
                           total_count = EXCLUDED.total_count,
                           updated_at  = EXCLUDED.updated_at""",
                    chat_id, message_id, emoji, new_total, now,
                )
            for emoji in previous:
                if emoji not in new_counts:
                    await conn.execute(
                        """INSERT INTO message_reaction_counts
                               (chat_id, message_id, emoji, total_count, updated_at)
                           VALUES ($1, $2, $3, 0, $4)
                           ON CONFLICT (chat_id, message_id, emoji) DO UPDATE SET
                               total_count = 0,
                               updated_at  = EXCLUDED.updated_at""",
                        chat_id, message_id, emoji, now,
                    )
    return deltas
