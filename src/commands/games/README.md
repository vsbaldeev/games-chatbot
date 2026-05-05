Emoji duel: real-time inline button race between two chat members.

DuelManager is a module-level singleton that owns all pending picks, acceptance
state, and active duels. Module-level wrappers in __init__.py expose cmd_duel and
handle_duel_callback for registration in bot.py.

## Lifecycle

```
/duel
    └── challenger picks opponent from inline keyboard (duel_pick)
            └── acceptance prompt sent to opponent (duel_accept / duel_reject)
                    ├── rejected → cancelled message
                    └── accepted → fire message with 🔫 button for both players
                                        └── first tap wins → duel_wins incremented
                                                              → check_new → notify_unlocks
```

## Timeouts

```
5 minutes  — acceptance timeout (auto-cancel if opponent does not respond)
5 minutes  — fire timeout (auto-cancel if neither player taps)
```

## Callbacks

```
duel_pick_<user_id>    — challenger selects opponent
duel_accept            — opponent accepts
duel_reject            — opponent declines
duel_fire              — first to tap wins
```
