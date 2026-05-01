"""
D&D game package.

Thin wrappers re-expose the public API that bot.py imports so that all
existing call sites work without changes:
  from src import dnd
  app.add_handler(CommandHandler("dnd_pvp", dnd.cmd_dnd_pvp, ...))
  app.add_handler(CallbackQueryHandler(dnd.handle_dnd_callback, pattern=dnd.DND_CALLBACK_PATTERN))
"""

from telegram import Update
from telegram.ext import ContextTypes

from src.commands.games.dnd.manager import dnd_manager
from src.commands.games.dnd.state import DND_CALLBACK_PATTERN

__all__ = [
    "cmd_dnd_pvp",
    "cmd_dnd_coop",
    "cmd_dnd_heist",
    "handle_dnd_callback",
    "DND_CALLBACK_PATTERN",
]


async def cmd_dnd_pvp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await dnd_manager.cmd_pvp(update, context)


async def cmd_dnd_coop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await dnd_manager.cmd_coop(update, context)


async def cmd_dnd_heist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await dnd_manager.cmd_heist(update, context)


async def handle_dnd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await dnd_manager.handle_callback(update, context)
