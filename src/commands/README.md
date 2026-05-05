Command handler implementations, organised by domain.

All handlers are registered in CommandHandlerManager (src/bot/handlers.py).
Adding a new command: handler → register in CommandHandlerManager → export from
src/commands/__init__.py → mention in cmd_help and the agent system prompt (src/agent.py).

## Commands

```
general.py
    /start          — welcome message
    /help           — full command list

statistics.py
    /achievements   — last 3 earned achievements with total count
    /top            — top-3 leaderboard by achievement count, with medal emojis

fun/
    /roast          — see src/commands/fun/README.md

games/
    /duel           — see src/commands/games/README.md
    /dnd_pvp
    /dnd_coop
    /dnd_heist      — see src/commands/games/dnd/README.md
```
