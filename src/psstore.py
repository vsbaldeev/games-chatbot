import logging
import sqlite3
import xml.etree.ElementTree as ET

import aiosqlite
import httpx

from src import config
from src import wishlist

logger = logging.getLogger(__name__)

# psdeals.net aggregates PS Store sales and provides a public RSS feed.
# If this URL stops working, check psdeals.net for the current feed path.
PSDEALS_RSS_URL = "https://psdeals.net/rss-feed"


async def init_sale_tracking() -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS announced_sales (
                chat_id      INTEGER NOT NULL,
                sale_title   TEXT NOT NULL,
                announced_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (chat_id, sale_title)
            )
        """)
        # Purge entries older than 7 days to prevent re-announcing long-running sales
        await db.execute(
            "DELETE FROM announced_sales WHERE announced_at < datetime('now', '-7 days')"
        )
        await db.commit()


async def fetch_current_sales() -> list[str]:
    """Fetch sale titles from psdeals.net RSS. Returns empty list on failure."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(
                PSDEALS_RSS_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
        root = ET.fromstring(response.text)
        return [
            item.text.strip()
            for item in root.findall(".//item/title")
            if item.text
        ]
    except Exception as error:
        logger.error(f"Failed to fetch PS Store sales from psdeals.net: {error}")
        return []


def __find_matching_title(game_name: str, sale_titles: list[str]) -> str | None:
    needle = game_name.lower()
    for title in sale_titles:
        if needle in title.lower():
            return title
    return None


async def __was_announced(chat_id: int, sale_title: str) -> bool:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT 1 FROM announced_sales WHERE chat_id = ? AND sale_title = ?",
            (chat_id, sale_title),
        )
        return await cursor.fetchone() is not None


async def __mark_announced(chat_id: int, sale_title: str) -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        try:
            await db.execute(
                "INSERT OR IGNORE INTO announced_sales (chat_id, sale_title) VALUES (?, ?)",
                (chat_id, sale_title),
            )
            await db.commit()
        except sqlite3.Error as err:
            logger.error(f"Failed to mark sale as announced: {err}")


async def find_wishlist_sales() -> dict[int, list[tuple[int, str, str, str]]]:
    """
    Returns {chat_id: [(user_id, username, wished_game, sale_title), ...]}
    Only includes chats where at least one wishlist game is currently on sale
    and has not been announced in the last 7 days.
    """
    sale_titles = await fetch_current_sales()
    if not sale_titles:
        return {}

    wishlists_by_chat = await wishlist.get_all_wishlists_by_chat()
    result: dict[int, list[tuple[int, str, str, str]]] = {}

    for chat_id, entries in wishlists_by_chat.items():
        matches = []
        for user_id, username, game_name in entries:
            matched_title = __find_matching_title(game_name, sale_titles)
            if matched_title and not await __was_announced(chat_id, matched_title):
                matches.append((user_id, username, game_name, matched_title))
        if matches:
            result[chat_id] = matches
            for unused_user_id, unused_username, unused_wished, sale_title in matches:
                await __mark_announced(chat_id, sale_title)

    return result
