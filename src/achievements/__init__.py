"""
Achievement system package.

Exposes the same public API as the old monolithic achievements.py so that
all callers (bot.py, handlers.py, commands.py, jobs.py, etc.) continue
to work with zero import-path changes.
"""

from src.achievements.definitions import (
    Achievement,
    ALL_ACHIEVEMENTS,
    ACHIEVEMENT_MAP,
    ACHIEVEMENT_RULES,
    SILENCE_THRESHOLDS,
    TRACKABLE_STATS,
    MAX_TRACKABLE_STATS,
)
from src.achievements.store import (
    init_tables,
    register_member,
    get_chat_members,
    get_all_chat_ids,
    increment_stat,
    update_max_stat,
    get_user_stats,
    get_announced_keys,
    mark_and_get_new,
    set_message_author,
    get_message_author,
    apply_reaction_counts,
)
from src.achievements.checker import (
    compute_earned,
    check_new_achievements,
    check_silence_achievements,
    get_user_achievements,
    get_chat_achievements_summary,
)

__all__ = [
    # definitions
    "Achievement",
    "ALL_ACHIEVEMENTS",
    "ACHIEVEMENT_MAP",
    "ACHIEVEMENT_RULES",
    "SILENCE_THRESHOLDS",
    "TRACKABLE_STATS",
    "MAX_TRACKABLE_STATS",
    # store
    "init_tables",
    "register_member",
    "get_chat_members",
    "get_all_chat_ids",
    "increment_stat",
    "update_max_stat",
    "get_user_stats",
    "get_announced_keys",
    "mark_and_get_new",
    "set_message_author",
    "get_message_author",
    "apply_reaction_counts",
    # checker
    "compute_earned",
    "check_new_achievements",
    "check_silence_achievements",
    "get_user_achievements",
    "get_chat_achievements_summary",
]
