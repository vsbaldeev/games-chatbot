Emoji duel: real-time inline button race between two chat members.

DuelManager is a module-level singleton that owns all pending picks, acceptance
state, and active duels. Module-level wrappers in __init__.py expose cmd_duel and
handle_duel_callback for registration in bot.py.

## Lifecycle

```
/duel
    │
    ▼
┌─────────────────────────────┐
│ @caller, выбери соперника:  │
│  [@vasya]  [@petya]         │  ← picker (60s timeout)
└─────────────────────────────┘
    │ caller picks @petya
    ▼ (same message, edited in-place)
┌──────────────────────────────────┐
│ 🔫 @caller бросает вызов @petya! │
│ @petya, ответишь?                │
│                                  │
│ ⏱ Осталось 60 сек.              │
│  [✅ Принять]  [❌ Отклонить]    │  ← every 5s only the number updates
└──────────────────────────────────┘
    │
    ├── @petya clicks ❌ / timeout
    │       ▼
    │   ┌──────────────────────────────┐
    │   │ 🏳️ @petya отклонил вызов    │
    │   │ @caller. Трус.               │
    │   └──────────────────────────────┘
    │
    └── @petya clicks ✅
            ▼ (same message, edited)
        ┌──────────────────────┐
        │ ⚔️ @caller vs @petya │
        │                      │
        │         [🔫]         │  ← both players can tap (5 min timeout)
        └──────────────────────┘
            │ first tap wins
            ▼ (same message, edited)
        ┌────────────────────────────────────────┐
        │ ⚔️ @caller vs @petya                   │
        │                                        │
        │ 💥 @caller попал в @petya! (1.23 сек)  │
        └────────────────────────────────────────┘
```

## Shot outcomes

```
60%  hit   — shooter wins
20%  self  — shooter hits themselves, opponent wins
20%  miss  — duel continues, opponent gets to shoot
```

## Timeouts

```
60s  — picker: auto-cancel if challenger does not pick a target
60s  — acceptance: countdown visible every 5s, auto-cancel if no response
5m   — fire: auto-cancel if neither player taps
```

## Callbacks

```
duel_pick:<index>  — challenger selects opponent from inline keyboard
duel_accept        — opponent accepts the challenge
duel_reject        — opponent declines
duel_fire          — first to tap wins
```
