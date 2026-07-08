Achievement system: stat tracking, rule evaluation, and unlock announcements.

Only **duel** achievements remain. The passive-counter achievements
(photos/videos/voice/stickers/forwards/night/essays) and the silence achievements
were retired — they tallied behaviour and mocked absence instead of driving
engagement. The stat **counters** are still tracked (see TRACKABLE_STATS): the
autonomous comedian consumes them via `src/agent/roast_material.py`.

TRACKABLE_STATS and MAX_TRACKABLE_STATS act as allowlists — increment_stat() and
update_max_stat() raise ValueError for any column not in those sets.

## Rules

```python
# ACHIEVEMENT_RULES: (stat_column, thresholds, achievement_keys)
("duel_wins", [10, 50, 100], ["duel_win_10", "duel_win_50", "duel_win_100"])
```

## Check flow

```
duel win → increment_stat("duel_wins")
    └── achievements.check_new_achievements(user_id, chat_id)
            └── for each rule in ACHIEVEMENT_RULES:
                    if stat >= threshold and key not in announced_achievements:
                        announce + insert into announced_achievements
```

`notify_unlocks` is invoked only from the duel flow
(`src/commands/games/duel.py`); message/reaction events increment counters but no
longer trigger unlock checks.

## Modules

```
definitions.py  Achievement dataclass, ALL_ACHIEVEMENTS catalogue (duel only), ACHIEVEMENT_RULES
store.py        DB ops: register_member, increment_stat, update_max_stat,
                get_user_stats, get_chat_members, get_all_chat_ids, mark_and_get_new
checker.py      compute_earned, check_new_achievements, notify_unlocks
```
