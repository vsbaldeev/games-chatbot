"""Persistence for sent memes — prevents repeats per chat."""

from src.store import db as database


async def init_table() -> None:
    async with database.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sent_memes (
                chat_id BIGINT NOT NULL,
                url     TEXT   NOT NULL,
                PRIMARY KEY (chat_id, url)
            )
        """)


async def get_seen_urls(chat_id: int) -> set[str]:
    async with database.acquire() as conn:
        rows = await conn.fetch(
            "SELECT url FROM sent_memes WHERE chat_id = $1",
            chat_id,
        )
    return {row["url"] for row in rows}


async def mark_seen(chat_id: int, url: str) -> None:
    async with database.acquire() as conn:
        await conn.execute(
            "INSERT INTO sent_memes (chat_id, url) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            chat_id, url,
        )
