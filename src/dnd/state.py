"""
Dataclasses and constants shared across the dnd package.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DND_JOIN_CALLBACK = "dnd_join"
DND_ACTION_CALLBACK_PREFIX = "dnd_a"
DND_CALLBACK_PATTERN = r"^dnd_"

DND_MODEL = "llama-3.3-70b-versatile"
DND_MIN_PLAYERS = 3
DND_ACTION_TIMEOUT = 45
DND_LOBBY_TIMEOUT = 300
DND_LLM_TIMEOUT = 30

# Sentinel user_id — no real Telegram user has ID 0.
DND_BOT_PLAYER_ID = 0
DND_BOT_PLAYER_NAME = "ДнД-Бот"


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LobbyState:
    """Transient state for a game lobby waiting for players to join."""

    chat_id: int
    message_id: int
    initiator_id: int
    players: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class ActiveGame:
    """State for a running D&D game from first round to resolution."""

    chat_id: int
    message_id: int
    scenario: str
    actions: list[str]
    players: list[tuple[int, str]]
    max_rounds: int = 3
    round_number: int = 1
    mode: str = "heist"       # "pvp" | "coop" | "heist"
    boss_name: str = ""
    boss_hp: int = 0
    boss_max_hp: int = 0
    choices: dict[int, int] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
