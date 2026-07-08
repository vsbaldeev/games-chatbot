"""
Achievement business logic: computing earned achievements, checking for new ones,
and notifying the chat on unlock.

Only duel achievements remain; the passive-counter and silence achievements were
retired, so the silence sweep and chat/user summary helpers are gone with them.
"""

from telegram.error import BadRequest
from telegram.ext import ContextTypes

from src import log
from src.achievements.definitions import (
    Achievement,
    ACHIEVEMENT_MAP,
    ACHIEVEMENT_RULES,
)
from src.achievements.store import (
    get_user_stats,
    mark_and_get_new,
)

logger = log.get_logger(__name__)


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


async def notify_unlocks(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    username: str,
) -> None:
    """Check for newly earned achievements and announce each one in the chat."""
    try:
        new_ach = await check_new_achievements(user_id, chat_id, username)
        for ach in new_ach:
            text = (
                f"🏆 @{username} получил достижение!\n\n"
                f"{ach.emoji} <b>{ach.title}</b>\n<i>{ach.description}</i>"
            )
            try:
                await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
            except BadRequest:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🏆 {username}: {ach.emoji} {ach.title} — {ach.description}",
                )
    except Exception as error:
        logger.warning("Achievement notification failed for user %s in chat %s: %s", user_id, chat_id, error)
