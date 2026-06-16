"""Bot command handlers."""

from src.commands.general import cmd_start, cmd_help
from src.commands.fun import cmd_meme
from src.commands.games import (
    cmd_duel,
    handle_duel_callback,
    DUEL_CALLBACK_PATTERN,
)
from src.commands.statistics import cmd_achievements, cmd_top

__all__ = [
    "cmd_start",
    "cmd_help",
    "cmd_meme",
    "cmd_duel",
    "handle_duel_callback",
    "DUEL_CALLBACK_PATTERN",
    "cmd_achievements",
    "cmd_top",
]
