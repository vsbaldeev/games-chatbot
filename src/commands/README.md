Command handler implementations, organised by domain.

All handlers are registered in CommandHandlerManager (src/bot/handlers.py).
Adding a new command: handler → register in CommandHandlerManager → export from
src/commands/__init__.py → mention in cmd_help and the agent system prompt (src/agent.py).

## Commands

```
general.py
    /start          — welcome message
    /help           — full command list

fun/
    /meme           — see src/commands/fun/README.md
    (roast generation lives here too — no command and no automatic trigger;
     kept as a reusable generator)

games/
    /duel           — see src/commands/games/README.md
```
