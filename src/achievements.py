import sqlite3
from dataclasses import dataclass

import aiosqlite

from src import config

TRACKABLE_STATS = {
    "crossplay_queries",
    "explain_queries",
    "night_messages",
    "research_queries",
    "coop_queries",
    "play_polls_created",
    "sale_notifications",
}

# New columns added after initial deploy — migrated safely in init_tables()
MIGRATION_COLUMNS = [
    "research_queries    INTEGER NOT NULL DEFAULT 0",
    "coop_queries        INTEGER NOT NULL DEFAULT 0",
    "play_polls_created  INTEGER NOT NULL DEFAULT 0",
    "sale_notifications  INTEGER NOT NULL DEFAULT 0",
]


@dataclass(frozen=True)
class Achievement:
    key: str
    emoji: str
    title: str
    description: str


ALL_ACHIEVEMENTS = [
    Achievement(
        "crossplay_paranoid", "🔍",
        "ПК-шник в душе",
        "Спрашивал про кросплей 3+ раз. Очевидно, надеялся, что ответ изменится.",
    ),
    Achievement(
        "explain_noob", "📚",
        "Гугл сломан",
        "Просил объяснить термин 3+ раз. FPS — это frames per second. Нет, не та игра.",
    ),
    Achievement(
        "night_owl", "🦉",
        "Сова в отрицании",
        "Писал боту между полуночью и 5 утра. Режим дня уехал без обратного адреса.",
    ),
    Achievement(
        "chronic_night_owl", "🌑",
        "Ночная легенда",
        "5+ ночных сообщений. Солнце для него — просто слух. Шторы куплены, не открывались.",
    ),
    Achievement(
        "night_creature", "🦇",
        "Летучая мышь чата",
        "10+ ночных сообщений. Врач бы поставил диагноз, но он тоже спит.",
    ),
    Achievement(
        "hoarder", "📦",
        "Коллекционер несыгранного",
        "5+ игр в вишлисте. Купить — не значит играть. Это инвестиция в вечность.",
    ),
    Achievement(
        "mega_hoarder", "🏗️",
        "Вишлист — это крик о помощи",
        "10+ игр. Библиотека строится, пыль копится, совесть молчит.",
    ),
    Achievement(
        "veteran", "🏅",
        "Зависимость подтверждена",
        "20+ обращений к боту. Диагноз поставлен, лечение не предусмотрено.",
    ),
    Achievement(
        "legend", "💀",
        "Клинический случай",
        "50+ обращений. Бот знает о нём больше, чем врач. И мама.",
    ),
    Achievement(
        "analyst", "🧠",
        "Диванный эксперт",
        "5+ игровых исследований. Зачем Гугл, когда есть бот и самомнение.",
    ),
    Achievement(
        "coop_evangelist", "👥",
        "Социофоб с кооп-зависимостью",
        "3+ раза искал кооп-игры. Людей ненавидит — но в одиночку скучно.",
    ),
    Achievement(
        "sale_hunter", "💸",
        "Скидка — смысл жизни",
        "Игра из вишлиста вышла на распродажу. Купит. Не сыграет. Классика жанра.",
    ),
]

ACHIEVEMENT_MAP = {achievement.key: achievement for achievement in ALL_ACHIEVEMENTS}


async def init_tables() -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_members (
                chat_id  INTEGER NOT NULL,
                user_id  INTEGER NOT NULL,
                username TEXT,
                PRIMARY KEY (chat_id, user_id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id              INTEGER NOT NULL,
                chat_id              INTEGER NOT NULL,
                username             TEXT,
                crossplay_queries    INTEGER NOT NULL DEFAULT 0,
                explain_queries      INTEGER NOT NULL DEFAULT 0,
                night_messages       INTEGER NOT NULL DEFAULT 0,
                total_interactions   INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, chat_id)
            )
        """)
        # Migrate: add new columns to existing tables without losing data
        for column_def in MIGRATION_COLUMNS:
            try:
                await db.execute(f"ALTER TABLE user_stats ADD COLUMN {column_def}")
            except sqlite3.OperationalError as err:
                if "duplicate column" not in str(err).lower():
                    raise
        await db.commit()


async def register_member(chat_id: int, user_id: int, username: str) -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO chat_members (chat_id, user_id, username) VALUES (?, ?, ?)",
            (chat_id, user_id, username),
        )
        await db.commit()


async def get_chat_members(chat_id: int) -> list[tuple[int, str]]:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT user_id, username FROM chat_members WHERE chat_id = ?",
            (chat_id,),
        )
        rows = await cursor.fetchall()
    return [(row[0], row[1] or f"user_{row[0]}") for row in rows]


async def get_all_chat_ids() -> list[int]:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute("SELECT DISTINCT chat_id FROM chat_members")
        rows = await cursor.fetchall()
    return [row[0] for row in rows]


async def increment_stat(user_id: int, chat_id: int, username: str, stat: str) -> None:
    if stat not in TRACKABLE_STATS:
        raise ValueError(f"Unknown stat '{stat}'. Allowed: {TRACKABLE_STATS}")
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute(
            f"""INSERT INTO user_stats (user_id, chat_id, username, {stat}, total_interactions)
                VALUES (?, ?, ?, 1, 1)
                ON CONFLICT(user_id, chat_id) DO UPDATE SET
                    {stat}               = {stat} + 1,
                    total_interactions   = total_interactions + 1,
                    username             = excluded.username""",
            (user_id, chat_id, username),
        )
        await db.commit()


async def increment_interaction(user_id: int, chat_id: int, username: str) -> None:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        await db.execute(
            """INSERT INTO user_stats (user_id, chat_id, username, total_interactions)
               VALUES (?, ?, ?, 1)
               ON CONFLICT(user_id, chat_id) DO UPDATE SET
                   total_interactions = total_interactions + 1,
                   username           = excluded.username""",
            (user_id, chat_id, username),
        )
        await db.commit()


async def get_user_stats(user_id: int, chat_id: int) -> dict[str, int]:
    return await __get_stats(user_id, chat_id)


async def __get_stats(user_id: int, chat_id: int) -> dict[str, int]:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            """SELECT crossplay_queries, explain_queries, night_messages, total_interactions,
                      research_queries, coop_queries, play_polls_created, sale_notifications
               FROM user_stats WHERE user_id = ? AND chat_id = ?""",
            (user_id, chat_id),
        )
        row = await cursor.fetchone()
    if not row:
        return {}
    return {
        "crossplay_queries":   row[0],
        "explain_queries":     row[1],
        "night_messages":      row[2],
        "total_interactions":  row[3],
        "research_queries":    row[4],
        "coop_queries":        row[5],
        "play_polls_created":  row[6],
        "sale_notifications":  row[7],
    }


async def __get_wishlist_count(user_id: int) -> int:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM wishlists WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
    return row[0] if row else 0


def __compute(stats: dict[str, int], wishlist_count: int) -> list[Achievement]:
    earned = []
    if stats.get("crossplay_queries", 0) >= 3:
        earned.append(ACHIEVEMENT_MAP["crossplay_paranoid"])
    if stats.get("explain_queries", 0) >= 3:
        earned.append(ACHIEVEMENT_MAP["explain_noob"])
    if stats.get("night_messages", 0) >= 1:
        earned.append(ACHIEVEMENT_MAP["night_owl"])
    if stats.get("night_messages", 0) >= 5:
        earned.append(ACHIEVEMENT_MAP["chronic_night_owl"])
    if stats.get("night_messages", 0) >= 10:
        earned.append(ACHIEVEMENT_MAP["night_creature"])
    if wishlist_count >= 5:
        earned.append(ACHIEVEMENT_MAP["hoarder"])
    if wishlist_count >= 10:
        earned.append(ACHIEVEMENT_MAP["mega_hoarder"])
    if stats.get("total_interactions", 0) >= 20:
        earned.append(ACHIEVEMENT_MAP["veteran"])
    if stats.get("total_interactions", 0) >= 50:
        earned.append(ACHIEVEMENT_MAP["legend"])
    if stats.get("research_queries", 0) >= 5:
        earned.append(ACHIEVEMENT_MAP["analyst"])
    if stats.get("coop_queries", 0) >= 3:
        earned.append(ACHIEVEMENT_MAP["coop_evangelist"])
    if stats.get("sale_notifications", 0) >= 1:
        earned.append(ACHIEVEMENT_MAP["sale_hunter"])
    return earned


async def get_user_achievements(user_id: int, chat_id: int) -> list[Achievement]:
    stats = await __get_stats(user_id, chat_id)
    wishlist_count = await __get_wishlist_count(user_id)
    return __compute(stats, wishlist_count)


async def get_chat_achievements_summary(chat_id: int) -> dict[str, list[Achievement]]:
    """Returns {username: [Achievement, ...]} for members with at least one achievement."""
    members = await get_chat_members(chat_id)
    result: dict[str, list[Achievement]] = {}
    for user_id, username in members:
        earned = await get_user_achievements(user_id, chat_id)
        if earned:
            result[username] = earned
    return result
