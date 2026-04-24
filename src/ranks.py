"""Rank system: points computed from existing user stats, displayed as funny tier titles."""

from dataclasses import dataclass

import aiosqlite

from src import achievements, config


@dataclass(frozen=True)
class Rank:
    title: str
    emoji: str
    min_points: int


# Ordered lowest → highest. get_rank() returns the highest tier the user qualifies for.
RANKS = [
    Rank("Только распаковал PS5",         "📦",   0),
    Rank("Читает инструкцию к джойстику", "📖",  15),
    Rank("Случайный казуал",              "🕹️",  35),
    Rank("Задрот с потенциалом",          "😤",  70),
    Rank("Чемпион дивана",                "🛋️", 120),
    Rank("Хардкорный нормис",             "⚡",  200),
    Rank("Профессиональный задрот",       "🏆",  300),
    Rank("Батя чата",                     "👑",  450),
]

# (stat_key, multiplier, human label for breakdown)
POINT_SOURCES = [
    ("total_interactions",  1, "взаимодействий с ботом"),
    ("night_messages",      3, "ночных подвигов"),
    ("play_polls_created",  5, "организованных сессий"),
    ("sale_notifications",  4, "найденных скидок"),
    ("research_queries",    2, "исследований"),
    ("coop_queries",        2, "кооп-запросов"),
    ("crossplay_queries",   1, "проверок кросплея"),
]


def pluralize_points(number: int) -> str:
    if 11 <= (number % 100) <= 14:
        return f"{number} очков"
    remainder = number % 10
    if remainder == 1:
        return f"{number} очко"
    if 2 <= remainder <= 4:
        return f"{number} очка"
    return f"{number} очков"


def compute_points(stats: dict[str, int], wishlist_count: int) -> int:
    return (
        sum(stats.get(stat, 0) * mult for stat, mult, _ in POINT_SOURCES)
        + wishlist_count  # +1 per wishlist item (collector bonus)
    )


def get_rank(points: int) -> Rank:
    current = RANKS[0]
    for rank in RANKS:
        if points >= rank.min_points:
            current = rank
    return current


def next_rank(points: int) -> Rank | None:
    """Returns the next tier above the current one, or None if already at max."""
    current = get_rank(points)
    for rank in RANKS:
        if rank.min_points > current.min_points:
            return rank
    return None


def build_breakdown(stats: dict[str, int], wishlist_count: int) -> list[str]:
    lines = []
    for stat, mult, label in POINT_SOURCES:
        value = stats.get(stat, 0)
        if value > 0:
            pts = value * mult
            lines.append(f"• {value} {label} × {mult} = {pluralize_points(pts)}")
    if wishlist_count > 0:
        lines.append(f"• {wishlist_count} игр в вишлисте × 1 = {pluralize_points(wishlist_count)}")
    return lines


async def __get_wishlist_count(user_id: int) -> int:
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM wishlists WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
    return row[0] if row else 0


async def get_user_rank_info(user_id: int, chat_id: int) -> tuple[int, Rank, list[str]]:
    stats = await achievements.get_user_stats(user_id, chat_id)
    wishlist_count = await __get_wishlist_count(user_id)
    points = compute_points(stats, wishlist_count)
    return points, get_rank(points), build_breakdown(stats, wishlist_count)


async def get_chat_leaderboard(chat_id: int) -> list[tuple[str, int, Rank]]:
    members = await achievements.get_chat_members(chat_id)
    board = []
    for user_id, username in members:
        stats = await achievements.get_user_stats(user_id, chat_id)
        wishlist_count = await __get_wishlist_count(user_id)
        points = compute_points(stats, wishlist_count)
        board.append((username, points, get_rank(points)))
    return sorted(board, key=lambda item: item[1], reverse=True)
