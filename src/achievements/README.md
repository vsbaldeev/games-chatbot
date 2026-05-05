Achievement system: stat tracking, rule evaluation, and unlock announcements.

TRACKABLE_STATS and MAX_TRACKABLE_STATS act as allowlists — increment_stat() and
update_max_stat() raise ValueError for any column not in those sets.

## Rules

```python
# ACHIEVEMENT_RULES: (stat_column, thresholds, achievement_keys)
("duel_wins",          [10,  50,  100],  ["duel_win_10",  "duel_win_50",  "duel_win_100"])
("photo_messages",     [10,  50,  100],  ["photo_10",     "photo_50",     "photo_100"])
("video_messages",     [10,  50,  100],  ["video_10",     "video_50",     "video_100"])
("voice_messages",     [10,  50,  100],  ["voice_10",     "voice_50",     "voice_100"])
("forwarded_messages", [10,  50,  100],  ["forward_10",   "forward_50",   "forward_100"])
("sticker_messages",   [10,  50,  100],  ["sticker_10",   "sticker_50",   "sticker_100"])
("night_messages",     [10,  50,  100],  ["night_10",     "night_50",     "night_100"])
("long_message_max",   [500, 1000, 2000],["essay_500",    "essay_1000",   "essay_2000"])

# SILENCE_THRESHOLDS: checked by silence_sweep_job, not on every message
(7,  "silence_7d")
(14, "silence_14d")
(30, "silence_30d")
```

## Check flow

```
increment_stat() / update_max_stat()
    └── achievements.check_new(chat_id, user_id)
            └── for each rule in ACHIEVEMENT_RULES:
                    if stat >= threshold and key not in announced_achievements:
                        announce + insert into announced_achievements
```

## Modules

```
definitions.py  Achievement dataclass, ALL_ACHIEVEMENTS catalogue, ACHIEVEMENT_RULES, SILENCE_THRESHOLDS
store.py        DB ops: init_tables, register_member, increment_stat, update_max_stat,
                get_user_stats, get_chat_members, get_all_chat_ids
checker.py      check_new, check_silence, get_achievement_summary, get_top_users
```
