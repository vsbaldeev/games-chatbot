"""Per-user game filter lists: banned (never suggest) and known (already plays)."""

import aiosqlite

from src import config

FILTER_BANNED = "banned"
FILTER_KNOWN = "known"


async def init_tables() -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS game_filters (
                user_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                game_name TEXT NOT NULL,
                filter_type TEXT NOT NULL CHECK(filter_type IN ('banned', 'known')),
                added_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, chat_id, game_name)
            )
        """)
        await db.commit()


async def set_filter(user_id: int, chat_id: int, game_name: str, filter_type: str) -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO game_filters (user_id, chat_id, game_name, filter_type)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, chat_id, game_name)
            DO UPDATE SET filter_type = excluded.filter_type, added_at = datetime('now')
            """,
            (user_id, chat_id, game_name, filter_type),
        )
        await db.commit()


async def remove_filter(user_id: int, chat_id: int, game_name: str) -> bool:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM game_filters WHERE user_id = ? AND chat_id = ? AND game_name = ? COLLATE NOCASE",
            (user_id, chat_id, game_name),
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_filters(user_id: int, chat_id: int) -> dict[str, list[str]]:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT game_name, filter_type FROM game_filters WHERE user_id = ? AND chat_id = ? ORDER BY game_name",
            (user_id, chat_id),
        )
        rows = await cursor.fetchall()
    result: dict[str, list[str]] = {FILTER_BANNED: [], FILTER_KNOWN: []}
    for game_name, filter_type in rows:
        result[filter_type].append(game_name)
    return result


def build_filter_hint(user_filters: dict[str, list[str]]) -> str | None:
    banned = user_filters.get(FILTER_BANNED, [])
    known = user_filters.get(FILTER_KNOWN, [])
    if not banned and not known:
        return None
    parts = []
    if banned:
        parts.append(f"[ФИЛЬТР] Никогда не предлагай этому пользователю: {', '.join(banned)}")
    if known:
        parts.append(f"[ФИЛЬТР] Пользователь уже знает и играет, не предлагай в общих рекомендациях: {', '.join(known)}")
    return "\n".join(parts)
