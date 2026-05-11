"""
Persistent store for every chat message regardless of media type.

Each row captures who sent what, whether it is a reply to another message,
and what kind of media it contains.  The table is the source of truth for
reply-chain resolution and conversation context assembly.
"""

import time

from src.store import db as database

MESSAGE_RETENTION_DAYS = 60

VOICE_PLACEHOLDER = "[voice]"
VIDEO_NOTE_PLACEHOLDER = "[video_note]"
VIDEO_PLACEHOLDER = "[video]"
PHOTO_PLACEHOLDER = "[photo]"
STICKER_PLACEHOLDER = "[sticker]"
ANIMATION_PLACEHOLDER = "[animation]"
AUDIO_PLACEHOLDER = "[audio]"

CHAIN_DEPTH_LIMIT = 10


def format_photo_content(caption: str | None) -> str:
    """Initial content for a photo: placeholder alone, or placeholder + caption on the next line.

    The placeholder prefix marks the row as still needing a vision description; the chain
    enricher uses needs_photo_description() to detect this state.
    """
    if caption:
        return f"{PHOTO_PLACEHOLDER}\n{caption}"
    return PHOTO_PLACEHOLDER


def needs_photo_description(content: str) -> bool:
    """True while the photo content is still in placeholder form (with or without a caption)."""
    return content == PHOTO_PLACEHOLDER or content.startswith(PHOTO_PLACEHOLDER + "\n")


def extract_photo_caption(content: str) -> str:
    """Pull the caption out of placeholder photo content. Returns empty string when none."""
    if content.startswith(PHOTO_PLACEHOLDER + "\n"):
        return content[len(PHOTO_PLACEHOLDER) + 1:]
    return ""


def combine_description_and_caption(description: str, caption: str) -> str:
    """Final enriched form: vision description, optionally suffixed with the original caption."""
    if caption:
        return f"{description}\n(подпись: {caption})"
    return description


def display_photo_content(content: str) -> str:
    """Strip placeholder prefix for prompt display when a row was never enriched."""
    if content.startswith(PHOTO_PLACEHOLDER + "\n"):
        return content[len(PHOTO_PLACEHOLDER) + 1:]
    if content == PHOTO_PLACEHOLDER:
        return ""
    return content


async def init_table() -> None:
    """Create the unified_messages table and its indexes if they do not exist."""
    async with database.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
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
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_unified_messages_chat_time
                ON unified_messages (chat_id, created_at DESC)
            """)
            await conn.execute("""
                ALTER TABLE unified_messages
                ADD COLUMN IF NOT EXISTS media_group_id TEXT
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_unified_messages_media_group
                ON unified_messages (chat_id, media_group_id)
                WHERE media_group_id IS NOT NULL
            """)


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
    media_group_id: str | None = None,
) -> None:
    """Insert a new message row. Silently ignores duplicate (chat_id, message_id) pairs."""
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO unified_messages
                (message_id, chat_id, user_id, username, content,
                 media_type, reply_to_msg_id, file_id, media_group_id, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (chat_id, message_id) DO NOTHING
            """,
            message_id, chat_id, user_id, username, content,
            media_type, reply_to_msg_id, file_id, media_group_id, time.time(),
        )


async def update_content(*, chat_id: int, message_id: int, content: str) -> None:
    """Replace the content of an existing row (e.g. swap a placeholder for a transcript)."""
    async with database.acquire() as conn:
        await conn.execute(
            "UPDATE unified_messages SET content = $1 WHERE chat_id = $2 AND message_id = $3",
            content, chat_id, message_id,
        )


async def get_chain(*, chat_id: int, message_id: int) -> list[dict]:
    """
    Walk the reply_to_msg_id chain upward from the given message and return
    all rows oldest-first.

    Stops after CHAIN_DEPTH_LIMIT hops to prevent runaway queries on deep threads.
    Returns an empty list if the root message is not found.
    """
    chain: list[dict] = []
    current_id: int | None = message_id
    async with database.acquire() as conn:
        for _ in range(CHAIN_DEPTH_LIMIT):
            if current_id is None:
                break
            row = await conn.fetchrow(
                """
                SELECT message_id, user_id, username, content, media_type,
                       reply_to_msg_id, file_id, media_group_id
                FROM unified_messages
                WHERE chat_id = $1 AND message_id = $2
                """,
                chat_id, current_id,
            )
            if row is None:
                break
            chain.append(dict(row))
            current_id = row["reply_to_msg_id"]

    chain.reverse()
    return chain


async def get_media_group(*, chat_id: int, media_group_id: str) -> list[dict]:
    """Return all messages belonging to a media group, ordered by message_id ascending."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT message_id, user_id, username, content, media_type,
                   reply_to_msg_id, file_id, media_group_id
            FROM unified_messages
            WHERE chat_id = $1 AND media_group_id = $2
            ORDER BY message_id ASC
            """,
            chat_id, media_group_id,
        )
    return [dict(row) for row in rows]


async def get_user_messages(*, chat_id: int, username: str, limit: int = 40) -> list[str]:
    """Return up to `limit` recent non-placeholder messages from a specific user, newest-first."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content FROM unified_messages
            WHERE chat_id = $1
              AND username = $2
              AND media_type IN ('text', 'voice', 'video_note', 'video')
              AND content NOT LIKE '[%]'
              AND content != ''
            ORDER BY created_at DESC
            LIMIT $3
            """,
            chat_id, username, limit,
        )
    return [row["content"] for row in rows]


async def cleanup_old(*, days: int = MESSAGE_RETENTION_DAYS) -> int:
    """Delete rows older than `days` days. Returns the number of deleted rows."""
    cutoff = time.time() - days * 86400
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM unified_messages WHERE created_at < $1",
            cutoff,
        )
    return int(result.split()[-1])


async def get_recent(*, chat_id: int, limit: int = 20) -> list[dict]:
    """Return the most recent messages for a chat, newest-first."""
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT message_id, user_id, username, content, media_type, created_at
            FROM unified_messages
            WHERE chat_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            chat_id, limit,
        )
    return [dict(row) for row in rows]
