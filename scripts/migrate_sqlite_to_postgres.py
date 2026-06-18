#!/usr/bin/env python3
"""
One-time data migration: SQLite → PostgreSQL.

Usage:
    python scripts/migrate_sqlite_to_postgres.py <sqlite_path> <postgres_url>

Example (inside the bot container on the VPS):
    python scripts/migrate_sqlite_to_postgres.py \\
        /sqlitedata/chat_history.db \\
        "postgresql://chatbot:password@postgres:5432/chatbot"

Safe to re-run: all inserts use ON CONFLICT DO NOTHING.
"""

import asyncio
import sqlite3
import sys

import asyncpg


SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_members (
    chat_id  BIGINT NOT NULL,
    user_id  BIGINT NOT NULL,
    username TEXT,
    PRIMARY KEY (chat_id, user_id)
);

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
);

CREATE TABLE IF NOT EXISTS announced_achievements (
    user_id BIGINT NOT NULL,
    chat_id BIGINT NOT NULL,
    key     TEXT   NOT NULL,
    PRIMARY KEY (user_id, chat_id, key)
);

CREATE TABLE IF NOT EXISTS message_authors (
    chat_id    BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    user_id    BIGINT NOT NULL,
    username   TEXT   NOT NULL,
    created_at BIGINT NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);

CREATE TABLE IF NOT EXISTS message_reaction_counts (
    chat_id     BIGINT  NOT NULL,
    message_id  BIGINT  NOT NULL,
    emoji       TEXT    NOT NULL,
    total_count INTEGER NOT NULL,
    updated_at  BIGINT  NOT NULL,
    PRIMARY KEY (chat_id, message_id, emoji)
);

CREATE TABLE IF NOT EXISTS unified_messages (
    message_id      BIGINT           NOT NULL,
    chat_id         BIGINT           NOT NULL,
    user_id         BIGINT           NOT NULL,
    username        TEXT             NOT NULL,
    content         TEXT             NOT NULL,
    media_type      TEXT             NOT NULL DEFAULT 'text',
    reply_to_msg_id BIGINT,
    file_id         TEXT,
    created_at      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_unified_messages_chat_time
    ON unified_messages (chat_id, created_at DESC);

CREATE TABLE IF NOT EXISTS user_memories (
    id         BIGSERIAL        PRIMARY KEY,
    chat_id    BIGINT           NOT NULL,
    user_id    BIGINT           NOT NULL,
    username   TEXT             NOT NULL,
    fact       TEXT             NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_memories_lookup
    ON user_memories (chat_id, user_id);

CREATE TABLE IF NOT EXISTS sent_memes (
    chat_id BIGINT NOT NULL,
    url     TEXT   NOT NULL,
    PRIMARY KEY (chat_id, url)
);

CREATE TABLE IF NOT EXISTS message_store (
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT,
    message    TEXT
);
"""


def get_sqlite_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone() is not None


async def create_schema(pg: asyncpg.Pool) -> None:
    print("Creating PostgreSQL schema...")
    async with pg.acquire() as conn:
        async with conn.transaction():
            for statement in SCHEMA.split(";"):
                statement = statement.strip()
                if statement:
                    await conn.execute(statement)
    print("  Done.")


async def copy_table(
    sqlite_conn: sqlite3.Connection,
    pg: asyncpg.Pool,
    table: str,
    columns: list[str],
) -> int:
    if not sqlite_table_exists(sqlite_conn, table):
        print(f"  {table}: not found in SQLite, skipping")
        return 0

    # Only select columns that actually exist in this SQLite DB
    # (guards against old DBs that predate some ALTER TABLE migrations)
    existing = set(get_sqlite_columns(sqlite_conn, table))
    safe_columns = [col for col in columns if col in existing]
    missing = set(columns) - existing
    if missing:
        print(f"  {table}: columns missing in SQLite (will use defaults): {missing}")

    cursor = sqlite_conn.execute(f"SELECT {', '.join(safe_columns)} FROM {table}")
    rows = cursor.fetchall()
    if not rows:
        print(f"  {table}: 0 rows")
        return 0

    placeholders = ", ".join(f"${idx + 1}" for idx in range(len(safe_columns)))
    col_list = ", ".join(safe_columns)
    sql = (
        f"INSERT INTO {table} ({col_list}) "
        f"VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    )
    async with pg.acquire() as conn:
        await conn.executemany(sql, [tuple(row) for row in rows])

    print(f"  {table}: {len(rows)} rows")
    return len(rows)


async def copy_user_memories(sqlite_conn: sqlite3.Connection, pg: asyncpg.Pool) -> int:
    if not sqlite_table_exists(sqlite_conn, "user_memories"):
        print("  user_memories: not found in SQLite, skipping")
        return 0

    cursor = sqlite_conn.execute(
        "SELECT id, chat_id, user_id, username, fact, updated_at FROM user_memories"
    )
    rows = cursor.fetchall()
    if not rows:
        print("  user_memories: 0 rows")
        return 0

    sql = (
        "INSERT INTO user_memories (id, chat_id, user_id, username, fact, updated_at) "
        "VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING"
    )
    async with pg.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(sql, [tuple(row) for row in rows])
            # Reset sequence so future inserts don't collide with copied IDs
            await conn.execute(
                "SELECT setval('user_memories_id_seq', "
                "(SELECT COALESCE(MAX(id), 1) FROM user_memories))"
            )

    print(f"  user_memories: {len(rows)} rows")
    return len(rows)


async def copy_message_store(sqlite_conn: sqlite3.Connection, pg: asyncpg.Pool) -> int:
    if not sqlite_table_exists(sqlite_conn, "message_store"):
        print("  message_store: not found in SQLite, skipping")
        return 0

    cursor = sqlite_conn.execute("SELECT id, session_id, message FROM message_store")
    rows = cursor.fetchall()
    if not rows:
        print("  message_store: 0 rows")
        return 0

    sql = (
        "INSERT INTO message_store (id, session_id, message) "
        "VALUES ($1, $2, $3) ON CONFLICT DO NOTHING"
    )
    async with pg.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(sql, [tuple(row) for row in rows])
            await conn.execute(
                "SELECT setval('message_store_id_seq', "
                "(SELECT COALESCE(MAX(id), 1) FROM message_store))"
            )

    print(f"  message_store: {len(rows)} rows")
    return len(rows)


async def migrate(sqlite_path: str, postgres_url: str) -> None:
    print(f"\nOpening SQLite: {sqlite_path}")
    sqlite_conn = sqlite3.connect(sqlite_path)

    print("Connecting to PostgreSQL...")
    pg = await asyncpg.create_pool(postgres_url)

    try:
        await create_schema(pg)
        print("\nCopying data...")

        simple_tables = [
            ("chat_members",            ["chat_id", "user_id", "username"]),
            ("user_stats",              ["user_id", "chat_id", "username",
                                         "laugh_reactions", "heart_reactions", "fire_reactions",
                                         "thumbsup_reactions", "emoji_messages", "sticker_messages",
                                         "forwarded_messages", "link_messages", "voice_messages",
                                         "video_messages", "video_note_messages", "photo_messages",
                                         "night_messages", "animation_messages", "roasted_count",
                                         "roulette_win_count", "duel_wins", "long_messages",
                                         "voice_max_duration", "long_message_max", "last_seen"]),
            ("announced_achievements",  ["user_id", "chat_id", "key"]),
            ("message_authors",         ["chat_id", "message_id", "user_id", "username", "created_at"]),
            ("message_reaction_counts", ["chat_id", "message_id", "emoji", "total_count", "updated_at"]),
            ("unified_messages",        ["message_id", "chat_id", "user_id", "username", "content",
                                         "media_type", "reply_to_msg_id", "file_id", "created_at"]),
            ("sent_memes",              ["chat_id", "url"]),
        ]

        for table, columns in simple_tables:
            await copy_table(sqlite_conn, pg, table, columns)

        await copy_user_memories(sqlite_conn, pg)
        await copy_message_store(sqlite_conn, pg)

        print("\nMigration complete. All data has been copied to PostgreSQL.")
    finally:
        sqlite_conn.close()
        await pg.close()


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/migrate_sqlite_to_postgres.py <sqlite_path> <postgres_url>")
        sys.exit(1)
    asyncio.run(migrate(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
