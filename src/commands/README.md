Command handler implementations, organised by domain.

All handlers are registered in CommandHandlerManager (src/app/handlers.py).
Adding a new command: handler → register in CommandHandlerManager → export from
src/commands/__init__.py → mention in cmd_help and the agent system prompt (src/agent.py).

## Commands

```
general.py
    /help           — full command list

games/
    /duel           — see src/commands/games/README.md
```
