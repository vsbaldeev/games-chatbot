"""
Async database operations for the achievement system.
"""

import time

import aiosqlite

from src.achievements.definitions import TRACKABLE_STATS, MAX_TRACKABLE_STATS
from src.store import db as database


async def __create_core_tables(db: aiosqlite.Connection) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS chat_members (
            chat_id  INTEGER NOT NULL,
            user_id  INTEGER NOT NULL,
            username TEXT,
            PRIMARY KEY (chat_id, user_id)
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id              INTEGER NOT NULL,
            chat_id              INTEGER NOT NULL,
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
            last_seen            INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, chat_id)
        )
    """)


async def __create_event_tables(db: aiosqlite.Connection) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS announced_achievements (
            user_id INTEGER NOT NULL,
            chat_id INTEGER NOT NULL,
            key     TEXT NOT NULL,
            PRIMARY KEY (user_id, chat_id, key)
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS message_authors (
            chat_id    INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            user_id    INTEGER NOT NULL,
            username   TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (chat_id, message_id)
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS message_reaction_counts (
            chat_id     INTEGER NOT NULL,
            message_id  INTEGER NOT NULL,
            emoji       TEXT NOT NULL,
            total_count INTEGER NOT NULL,
            updated_at  INTEGER NOT NULL,
            PRIMARY KEY (chat_id, message_id, emoji)
        )
    """)


async def __run_migrations(db: aiosqlite.Connection) -> None:
    migrations = [
        "ALTER TABLE user_stats ADD COLUMN long_messages INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE user_stats ADD COLUMN voice_max_duration INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE user_stats ADD COLUMN long_message_max INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE user_stats ADD COLUMN last_seen INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE user_stats ADD COLUMN video_note_messages INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE user_stats ADD COLUMN video_messages INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE user_stats ADD COLUMN photo_messages INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE user_stats ADD COLUMN night_messages INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE user_stats ADD COLUMN duel_wins INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE user_stats ADD COLUMN animation_messages INTEGER NOT NULL DEFAULT 0",
    ]
    for migration in migrations:
        try:
            await db.execute(migration)
        except aiosqlite.OperationalError as error:
            if "duplicate column" not in str(error):
                raise


async def init_tables() -> None:
    """Create all achievement-related tables and run pending column migrations."""
    db = await database.get()
    await __create_core_tables(db)
    await __create_event_tables(db)
    await __run_migrations(db)
    await db.commit()


async def register_member(chat_id: int, user_id: int, username: str) -> None:
    """Add a user to chat_members if not already present (no-op on conflict)."""
    db = await database.get()
    await db.execute(
        "INSERT OR IGNORE INTO chat_members (chat_id, user_id, username) VALUES (?, ?, ?)",
        (chat_id, user_id, username),
    )
    await db.commit()


async def get_chat_members(chat_id: int) -> list[tuple[int, str]]:
    """Return (user_id, username) pairs for every member registered in this chat."""
    db = await database.get()
    cursor = await db.execute(
        "SELECT user_id, username FROM chat_members WHERE chat_id = ?",
        (chat_id,),
    )
    rows = await cursor.fetchall()
    return [(row[0], row[1] or f"user_{row[0]}") for row in rows]


async def get_all_chat_ids() -> list[int]:
    """Return all distinct chat IDs that have at least one registered member."""
    db = await database.get()
    cursor = await db.execute("SELECT DISTINCT chat_id FROM chat_members")
    rows = await cursor.fetchall()
    return [row[0] for row in rows]


async def increment_stat(user_id: int, chat_id: int, username: str, stat: str) -> None:
    """Increment a counter stat by 1, upserting the user_stats row."""
    if stat not in TRACKABLE_STATS:
        raise ValueError(f"Unknown stat '{stat}'. Allowed: {TRACKABLE_STATS}")
    db = await database.get()
    await db.execute(
        f"""INSERT INTO user_stats (user_id, chat_id, username, {stat}, last_seen)
            VALUES (?, ?, ?, 1, strftime('%s','now'))
            ON CONFLICT(user_id, chat_id) DO UPDATE SET
                {stat}    = {stat} + 1,
                username  = excluded.username,
                last_seen = strftime('%s','now')""",
        (user_id, chat_id, username),
    )
    await db.commit()


async def update_max_stat(user_id: int, chat_id: int, username: str, stat: str, value: int) -> None:
    """Update a max-tracking stat if value exceeds the stored maximum."""
    if stat not in MAX_TRACKABLE_STATS:
        raise ValueError(f"Unknown max stat '{stat}'. Allowed: {MAX_TRACKABLE_STATS}")
    db = await database.get()
    await db.execute(
        f"""INSERT INTO user_stats (user_id, chat_id, username, {stat}, last_seen)
            VALUES (?, ?, ?, ?, strftime('%s','now'))
            ON CONFLICT(user_id, chat_id) DO UPDATE SET
                {stat}    = MAX({stat}, excluded.{stat}),
                username  = excluded.username,
                last_seen = strftime('%s','now')""",
        (user_id, chat_id, username, value),
    )
    await db.commit()


async def get_user_stats(user_id: int, chat_id: int) -> dict[str, int]:
    """Return the full stat row for a user in a chat, or {} if no row exists."""
    db = await database.get()
    cursor = await db.execute(
        """SELECT laugh_reactions, heart_reactions, fire_reactions, thumbsup_reactions,
                  emoji_messages, sticker_messages, forwarded_messages,
                  link_messages, voice_messages, video_messages, video_note_messages,
                  photo_messages, night_messages, long_messages,
                  voice_max_duration, long_message_max,
                  roasted_count, roulette_win_count, duel_wins, animation_messages
           FROM user_stats WHERE user_id = ? AND chat_id = ?""",
        (user_id, chat_id),
    )
    row = await cursor.fetchone()
    if not row:
        return {}
    return {
        "laugh_reactions":     row[0],
        "heart_reactions":     row[1],
        "fire_reactions":      row[2],
        "thumbsup_reactions":  row[3],
        "emoji_messages":      row[4],
        "sticker_messages":    row[5],
        "forwarded_messages":  row[6],
        "link_messages":       row[7],
        "voice_messages":      row[8],
        "video_messages":      row[9],
        "video_note_messages": row[10],
        "photo_messages":      row[11],
        "night_messages":      row[12],
        "long_messages":       row[13],
        "voice_max_duration":  row[14],
        "long_message_max":    row[15],
        "roasted_count":       row[16],
        "roulette_win_count":  row[17],
        "duel_wins":           row[18],
        "animation_messages":  row[19],
    }


async def get_announced_keys(user_id: int, chat_id: int) -> set[str]:
    """Return all achievement keys already announced to this user in this chat."""
    db = await database.get()
    cursor = await db.execute(
        "SELECT key FROM announced_achievements WHERE user_id = ? AND chat_id = ?",
        (user_id, chat_id),
    )
    rows = await cursor.fetchall()
    return {row[0] for row in rows}


async def mark_and_get_new(user_id: int, chat_id: int, keys: list[str]) -> list[str]:
    """Insert keys into announced_achievements atomically; return only newly inserted keys."""
    if not keys:
        return []
    placeholders = ", ".join("(?, ?, ?)" for _ in keys)
    params = [value for key in keys for value in (user_id, chat_id, key)]
    db = await database.get()
    cursor = await db.execute(
        f"INSERT OR IGNORE INTO announced_achievements (user_id, chat_id, key) "
        f"VALUES {placeholders} RETURNING key",
        params,
    )
    rows = await cursor.fetchall()
    await db.commit()
    return [row[0] for row in rows]


async def set_message_author(
    chat_id: int, message_id: int, user_id: int, username: str
) -> None:
    """Record which user authored a message (for reaction credit lookups)."""
    db = await database.get()
    await db.execute(
        "INSERT OR IGNORE INTO message_authors "
        "(chat_id, message_id, user_id, username, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (chat_id, message_id, user_id, username, int(time.time())),
    )
    await db.commit()


async def get_message_author(chat_id: int, message_id: int) -> tuple[int, str] | None:
    """Return (user_id, username) for the author of a message, or None if unknown."""
    db = await database.get()
    cursor = await db.execute(
        "SELECT user_id, username FROM message_authors "
        "WHERE chat_id = ? AND message_id = ?",
        (chat_id, message_id),
    )
    row = await cursor.fetchone()
    return (row[0], row[1]) if row else None


async def apply_reaction_counts(
    chat_id: int, message_id: int, new_counts: dict[str, int]
) -> dict[str, int]:
    """Persist the latest per-emoji reaction totals and return positive deltas per emoji."""
    deltas: dict[str, int] = {}
    now = int(time.time())
    db = await database.get()
    cursor = await db.execute(
        "SELECT emoji, total_count FROM message_reaction_counts "
        "WHERE chat_id = ? AND message_id = ?",
        (chat_id, message_id),
    )
    previous = {row[0]: row[1] for row in await cursor.fetchall()}

    for emoji, new_total in new_counts.items():
        delta = max(0, new_total - previous.get(emoji, 0))
        if delta > 0:
            deltas[emoji] = delta
        await db.execute(
            "INSERT OR REPLACE INTO message_reaction_counts "
            "(chat_id, message_id, emoji, total_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (chat_id, message_id, emoji, new_total, now),
        )

    for emoji in previous:
        if emoji not in new_counts:
            await db.execute(
                "INSERT OR REPLACE INTO message_reaction_counts "
                "(chat_id, message_id, emoji, total_count, updated_at) "
                "VALUES (?, ?, ?, 0, ?)",
                (chat_id, message_id, emoji, now),
            )

    await db.commit()
    return deltas
