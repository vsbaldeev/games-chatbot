"""
Persistent cache of vision descriptions for stickers.

Users resend the same stickers constantly, so descriptions are cached keyed
by Telegram's ``file_unique_id`` — the identifier that stays stable across
resends and bots (unlike ``file_id``). Each distinct sticker costs exactly
one vision call for its lifetime; rows are tiny and never expire.
"""

import time

from src.store import db as database


async def get_description(file_unique_id: str) -> str | None:
    """Return the cached description for a sticker, or None when never described.

    Args:
        file_unique_id: Telegram's stable identifier of the sticker file.

    Returns:
        The cached vision description, or ``None`` on a cache miss.
    """
    async with database.acquire() as conn:
        return await conn.fetchval(
            "SELECT description FROM sticker_descriptions WHERE file_unique_id = $1",
            file_unique_id,
        )


async def save_description(file_unique_id: str, description: str) -> None:
    """Cache a sticker's vision description; a concurrent duplicate wins silently.

    Args:
        file_unique_id: Telegram's stable identifier of the sticker file.
        description: Non-empty vision description to cache.
    """
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sticker_descriptions (file_unique_id, description, created_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (file_unique_id) DO NOTHING
            """,
            file_unique_id, description, time.time(),
        )
