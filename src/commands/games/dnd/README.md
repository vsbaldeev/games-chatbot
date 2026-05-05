D&D text adventure: lobby → LLM-generated scenario → round resolution.

DndManager is a stateful singleton that owns all active lobbies and games.
ScenarioGenerator wraps the LLM calls. Three game modes share the same lobby
and round machinery; only scenario prompts and round counts differ.

## Game modes

```
/dnd_pvp    PvP adventure   1 round    all players act simultaneously; LLM picks a winner
/dnd_coop   Co-op           2 rounds   players act together vs. a boss NPC
/dnd_heist  The Great Heist 3 phases   infiltration → job → escape; progressive difficulty
```

## Lobby lifecycle

```
/dnd_*
    └── lobby opened (join button, DND_LOBBY_TIMEOUT)
            └── players join via inline button
                    └── game starts when creator taps Start (≥ DND_MIN_PLAYERS = 2)
                            └── ScenarioGenerator generates opening + first action choices
                                    └── each round: players pick action (DND_ACTION_TIMEOUT)
                                            └── LLM resolves round → next choices or final outcome
```

## Constants

```python
DND_MIN_PLAYERS      = 2
DND_LOBBY_TIMEOUT    = 300   # seconds
DND_ACTION_TIMEOUT   = 120   # seconds
DND_BOT_PLAYER_NAME  = "Таверна"   # NPC bot player for co-op boss
```

## LLM

```
Model:  llama-3.3-70b-versatile
Style:  short paragraphs, in Russian, D&D tone
Input:  game mode, player list, chosen actions per round
Output: narrative + next action options (or final resolution)
```
