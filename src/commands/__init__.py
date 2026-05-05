"""Bot command handlers."""

from src.commands.general import cmd_start, cmd_help
from src.commands.fun import cmd_roast
from src.commands.games import (
    cmd_dnd_pvp,
    cmd_dnd_coop,
    cmd_dnd_heist,
    handle_dnd_callback,
    DND_CALLBACK_PATTERN,
    cmd_duel,
    handle_duel_callback,
    DUEL_CALLBACK_PATTERN,
)
from src.commands.statistics import cmd_achievements, cmd_top

__all__ = [
    "cmd_start",
    "cmd_help",
    "cmd_roast",
    "cmd_dnd_pvp",
    "cmd_dnd_coop",
    "cmd_dnd_heist",
    "handle_dnd_callback",
    "DND_CALLBACK_PATTERN",
    "cmd_duel",
    "handle_duel_callback",
    "DUEL_CALLBACK_PATTERN",
    "cmd_achievements",
    "cmd_top",
]
