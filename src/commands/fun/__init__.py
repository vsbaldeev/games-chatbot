"""Fun commands — /roast and /roulette."""

from src.commands.fun.prozharka import cmd_roast
from src.commands.fun.roulette import cmd_roulette, russian_roulette

__all__ = ["cmd_roast", "cmd_roulette", "russian_roulette"]
