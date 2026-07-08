"""
Static definitions for the achievement system.

Contains the Achievement dataclass, the catalogue of achievements, the mapping
used for fast key-based lookup, and the rule table that ties stat thresholds to
achievement keys.

Only the duel achievements survive: the passive-counter and silence achievements
were retired because they tallied behaviour and mocked absence instead of driving
engagement. The stat counters themselves are still tracked (see TRACKABLE_STATS)
because the autonomous comedian and offence auto-roast consume them.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Stat categories
# ---------------------------------------------------------------------------

TRACKABLE_STATS = {
    "laugh_reactions",
    "heart_reactions",
    "fire_reactions",
    "thumbsup_reactions",
    "emoji_messages",
    "sticker_messages",
    "forwarded_messages",
    "link_messages",
    "voice_messages",
    "video_messages",
    "video_note_messages",
    "photo_messages",
    "night_messages",
    "animation_messages",
    "roasted_count",
    "duel_wins",
}

# "Max" stats — tracked with update_max_stat rather than increment_stat
MAX_TRACKABLE_STATS = {"voice_max_duration", "long_message_max"}


# ---------------------------------------------------------------------------
# Achievement dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Achievement:
    key: str
    emoji: str
    title: str
    description: str


# ---------------------------------------------------------------------------
# Full achievement catalogue
# ---------------------------------------------------------------------------

ALL_ACHIEVEMENTS: list[Achievement] = [
    # --- duel wins ---
    Achievement("duel_win_10",  "🔫", "Быстрый палец",
                "10 побед. Твоя реакция пугает. Ты либо профессиональный киберспортсмен на пенсии, либо просто не выпускаешь телефон из рук даже в душе."),
    Achievement("duel_win_50",  "🐎", "Самый быстрый на диване",
                "50 побед. Твоя скорость нажатия на пиксельный пистолет легендарна. Жаль, что этот навык не помогает в реальной жизни, но в этом болоте ты — король."),
    Achievement("duel_win_100", "🏆", "Кибер-кликун",
                "100 побед. Ты официально стёр отпечаток пальца об экран. Поздравляем, ты — абсолютный чемпион по бессмысленному насилию над эмодзи. Твои враги мертвы (цифрово), а твоё ЧСВ — в космосе."),
]


# ---------------------------------------------------------------------------
# Lookup structures
# ---------------------------------------------------------------------------

ACHIEVEMENT_MAP: dict[str, Achievement] = {ach.key: ach for ach in ALL_ACHIEVEMENTS}


# ---------------------------------------------------------------------------
# Rules: (stat_column, thresholds, achievement_keys)
# ---------------------------------------------------------------------------

# Each tuple maps a user_stats column to a list of (threshold, achievement_key) pairs.
# The lists are ordered ascending so checking stops at the first unmet threshold.
ACHIEVEMENT_RULES: list[tuple[str, list[int], list[str]]] = [
    ("duel_wins", [10, 50, 100], ["duel_win_10", "duel_win_50", "duel_win_100"]),
]
