"""
Achievement business logic: computing earned achievements, checking for new ones,
summarising what a user or chat has unlocked.
"""

import time

import aiosqlite

from src import config
from src.achievements.definitions import (
    Achievement,
    ACHIEVEMENT_MAP,
    ACHIEVEMENT_RULES,
    SILENCE_THRESHOLDS,
)
from src.achievements.store import (
    get_user_stats,
    get_chat_members,
    get_announced_keys,
    mark_and_get_new,
)


def compute_earned(stats: dict[str, int]) -> list[Achievement]:
    """Return all achievements earned by a user given their current stat dict."""
    earned = []
    for stat_name, thresholds, keys in ACHIEVEMENT_RULES:
        stat_value = stats.get(stat_name, 0)
        for threshold, key in zip(thresholds, keys):
            if stat_value >= threshold:
                earned.append(ACHIEVEMENT_MAP[key])
    return earned


async def check_new_achievements(user_id: int, chat_id: int, username: str) -> list[Achievement]:
    """Return achievements earned since the last call and mark them as announced."""
    stats = await get_user_stats(user_id, chat_id)
    earned = compute_earned(stats)
    new_keys = await mark_and_get_new(user_id, chat_id, [ach.key for ach in earned])
    return [ACHIEVEMENT_MAP[key] for key in new_keys]


async def check_silence_achievements(user_id: int, chat_id: int, username: str) -> list[Achievement]:
    """Return the next unawarded silence achievement (at most one per call) and mark it announced.

    Throttled to one per sweep to avoid flooding the chat when a long-inactive user
    is first seen. Called by the daily sweep job only — not on every message,
    since activity resets last_seen.
    """
    async with aiosqlite.connect(config.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT last_seen FROM user_stats WHERE user_id = ? AND chat_id = ?",
            (user_id, chat_id),
        )
        row = await cursor.fetchone()
    if not row or row[0] == 0:
        return []

    elapsed_days = (time.time() - row[0]) / 86400
    # SILENCE_THRESHOLDS is ordered ascending; stop at the first newly inserted key.
    for days, key in SILENCE_THRESHOLDS:
        if elapsed_days < days:
            break
        new_keys = await mark_and_get_new(user_id, chat_id, [key])
        if new_keys:
            return [ACHIEVEMENT_MAP[key]]
    return []


async def get_user_achievements(user_id: int, chat_id: int) -> list[Achievement]:
    """Return all achievements a user has earned (stat-based + silence-based)."""
    stats = await get_user_stats(user_id, chat_id)
    stat_earned = compute_earned(stats)
    announced_keys = await get_announced_keys(user_id, chat_id)
    silence_keys = {key for _, key in SILENCE_THRESHOLDS}
    silence_earned = [ACHIEVEMENT_MAP[key] for key in silence_keys if key in announced_keys]
    return stat_earned + silence_earned


async def get_chat_achievements_summary(chat_id: int) -> dict[str, list[Achievement]]:
    """Return {username: [Achievement, ...]} for every member with at least one achievement."""
    members = await get_chat_members(chat_id)
    result: dict[str, list[Achievement]] = {}
    for user_id, username in members:
        earned = await get_user_achievements(user_id, chat_id)
        if earned:
            result[username] = earned
    return result
