"""Persistence for sent memes — prevents repeats per chat."""

import aiosqlite

from src.store import db as database


async def init_table() -> None:
    db = await database.get()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS sent_memes (
            chat_id INTEGER NOT NULL,
            url     TEXT    NOT NULL,
            PRIMARY KEY (chat_id, url)
        )
    """)
    await db.commit()


async def get_seen_urls(chat_id: int) -> set[str]:
    db = await database.get()
    cursor = await db.execute(
        "SELECT url FROM sent_memes WHERE chat_id = ?",
        (chat_id,),
    )
    rows = await cursor.fetchall()
    return {row[0] for row in rows}


async def mark_seen(chat_id: int, url: str) -> None:
    db = await database.get()
    await db.execute(
        "INSERT OR IGNORE INTO sent_memes (chat_id, url) VALUES (?, ?)",
        (chat_id, url),
    )
    await db.commit()
