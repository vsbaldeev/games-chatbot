"""
Persistent store for every chat message regardless of media type.

Each row captures who sent what, whether it is a reply to another message,
and what kind of media it contains.  The table is the source of truth for
reply-chain resolution and conversation context assembly.
"""

import time

from src.store import db as database

MESSAGE_RETENTION_DAYS = 60

# Placeholder content stored immediately for media messages before the real
# transcription/description is available.
VOICE_PLACEHOLDER = "[voice]"
VIDEO_NOTE_PLACEHOLDER = "[video_note]"
VIDEO_PLACEHOLDER = "[video]"
PHOTO_PLACEHOLDER = "[photo]"

# Maximum number of hops to follow when resolving a reply chain.
CHAIN_DEPTH_LIMIT = 10


async def init_table() -> None:
    """Create the unified_messages table and its index if they do not exist."""
    db = await database.get()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS unified_messages (
            message_id      INTEGER NOT NULL,
            chat_id         INTEGER NOT NULL,
            user_id         INTEGER NOT NULL,
            username        TEXT    NOT NULL,
            content         TEXT    NOT NULL,
            media_type      TEXT    NOT NULL DEFAULT 'text',
            reply_to_msg_id INTEGER,
            file_id         TEXT,
            created_at      REAL    NOT NULL,
            PRIMARY KEY (chat_id, message_id)
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_unified_messages_chat_time
        ON unified_messages (chat_id, created_at DESC)
    """)
    await db.commit()


async def insert(
    *,
    chat_id: int,
    message_id: int,
    user_id: int,
    username: str,
    content: str,
    media_type: str = "text",
    reply_to_msg_id: int | None = None,
    file_id: str | None = None,
) -> None:
    """Insert a new message row.  Silently ignores duplicate (chat_id, message_id) pairs."""
    db = await database.get()
    await db.execute(
        """
        INSERT OR IGNORE INTO unified_messages
            (message_id, chat_id, user_id, username, content,
             media_type, reply_to_msg_id, file_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id, chat_id, user_id, username, content,
            media_type, reply_to_msg_id, file_id, time.time(),
        ),
    )
    await db.commit()


async def update_content(*, chat_id: int, message_id: int, content: str) -> None:
    """Replace the content of an existing row (e.g. swap a placeholder for a transcript)."""
    db = await database.get()
    await db.execute(
        "UPDATE unified_messages SET content = ? WHERE chat_id = ? AND message_id = ?",
        (content, chat_id, message_id),
    )
    await db.commit()


async def get_chain(*, chat_id: int, message_id: int) -> list[dict]:
    """
    Walk the reply_to_msg_id chain upward from the given message and return
    all rows oldest-first.

    Stops after CHAIN_DEPTH_LIMIT hops to prevent runaway queries on deep threads.
    Returns an empty list if the root message is not found.
    """
    chain: list[dict] = []
    current_id: int | None = message_id
    db = await database.get()

    for _ in range(CHAIN_DEPTH_LIMIT):
        if current_id is None:
            break
        cursor = await db.execute(
            """
            SELECT message_id, user_id, username, content, media_type, reply_to_msg_id, file_id
            FROM unified_messages
            WHERE chat_id = ? AND message_id = ?
            """,
            (chat_id, current_id),
        )
        row = await cursor.fetchone()
        if row is None:
            break
        chain.append(dict(row))
        current_id = row["reply_to_msg_id"]

    chain.reverse()
    return chain


async def cleanup_old(*, days: int = MESSAGE_RETENTION_DAYS) -> int:
    """Delete rows older than `days` days. Returns the number of deleted rows."""
    cutoff = time.time() - days * 86400
    db = await database.get()
    cursor = await db.execute(
        "DELETE FROM unified_messages WHERE created_at < ?",
        (cutoff,),
    )
    await db.commit()
    return cursor.rowcount


async def get_recent(*, chat_id: int, limit: int = 20) -> list[dict]:
    """Return the most recent messages for a chat, newest-first."""
    db = await database.get()
    rows = await db.execute_fetchall(
        """
        SELECT message_id, user_id, username, content, media_type, created_at
        FROM unified_messages
        WHERE chat_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (chat_id, limit),
    )
    return [dict(row) for row in rows]
