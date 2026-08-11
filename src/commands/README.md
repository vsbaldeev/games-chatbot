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
    (roast generation lives here — no command and no automatic trigger;
     kept as a reusable generator. /meme was retired: memes are asked for in
     words now and handled by the MEME_REQUEST filter verdict, see
     src/memes/README.md)

games/
    /duel           — see src/commands/games/README.md
```
