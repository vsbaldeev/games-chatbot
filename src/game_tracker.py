"""
Tracks which games have already been suggested per chat, to avoid repeating suggestions.
"""

import aiosqlite

from src import config


async def init_tables() -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS suggested_games (
                chat_id      INTEGER NOT NULL,
                game_name    TEXT    NOT NULL,
                game_type    TEXT    NOT NULL,
                suggested_at TEXT    NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (chat_id, game_name, game_type)
            )
        """)
        await db.commit()


async def mark_suggested(chat_id: int, game_name: str, game_type: str) -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO suggested_games (chat_id, game_name, game_type) VALUES (?, ?, ?)",
            (chat_id, game_name.lower().strip(), game_type),
        )
        await db.commit()


async def get_suggested(chat_id: int, game_type: str, limit: int = 50) -> list[str]:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT game_name FROM suggested_games "
            "WHERE chat_id = ? AND game_type = ? "
            "ORDER BY suggested_at DESC LIMIT ?",
            (chat_id, game_type, limit),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]
