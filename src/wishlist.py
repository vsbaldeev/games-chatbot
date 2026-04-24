import aiosqlite

from src import config


async def init_tables() -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS wishlists (
                user_id  INTEGER NOT NULL,
                chat_id  INTEGER NOT NULL,
                username TEXT,
                game_name TEXT NOT NULL,
                added_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, game_name)
            )
        """)
        await db.commit()


async def add_game(user_id: int, chat_id: int, username: str, game_name: str) -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO wishlists (user_id, chat_id, username, game_name) VALUES (?, ?, ?, ?)",
            (user_id, chat_id, username, game_name.strip()),
        )
        await db.commit()


async def remove_game(user_id: int, game_name: str) -> bool:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM wishlists WHERE user_id = ? AND LOWER(game_name) = LOWER(?)",
            (user_id, game_name.strip()),
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_user_wishlist(user_id: int) -> list[str]:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT game_name FROM wishlists WHERE user_id = ? ORDER BY added_at",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]


async def get_chat_wishlists(chat_id: int) -> dict[str, list[str]]:
    """Returns {username: [game1, game2, ...]} for all users in a chat."""
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT username, game_name FROM wishlists WHERE chat_id = ? ORDER BY username, added_at",
            (chat_id,),
        )
        rows = await cursor.fetchall()

    result: dict[str, list[str]] = {}
    for username, game_name in rows:
        display = username or "Аноним"
        result.setdefault(display, []).append(game_name)
    return result


async def get_all_wishlists_by_chat() -> dict[int, list[tuple[int, str, str]]]:
    """Returns {chat_id: [(user_id, username, game_name), ...]} across all chats."""
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute("SELECT chat_id, user_id, username, game_name FROM wishlists")
        rows = await cursor.fetchall()

    result: dict[int, list[tuple[int, str, str]]] = {}
    for chat_id, user_id, username, game_name in rows:
        result.setdefault(chat_id, []).append((user_id, username or "Аноним", game_name))
    return result
