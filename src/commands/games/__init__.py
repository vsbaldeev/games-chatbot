"""Game command handlers — duel."""

from src.commands.games.duel import (
    cmd_duel,
    handle_duel_callback,
    DUEL_CALLBACK_PATTERN,
)

__all__ = [
    "cmd_duel",
    "handle_duel_callback",
    "DUEL_CALLBACK_PATTERN",
]
