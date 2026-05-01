"""Bot command handlers."""

from telegram import Update
from telegram.ext import ContextTypes

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


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет. Я здесь. Спрашивайте про игры — если, конечно, есть что спросить."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет. Я здесь.\n\n"
        "Чтобы всё работало правильно:\n\n"
        "• Напишите что-нибудь в чат — каждый участник регистрируется, когда впервые пишет сообщение, реагирует или отправляет медиа. Только зарегистрированные попадают в рулетку и дуэли.\n"
        "Что умею:\n"
        "/dnd_pvp, /dnd_coop, /dnd_heist — D&D-приключение (PvP, кооп с боссом, великое ограбление)\n"
        "/duel — эмодзи-дуэль между двумя участниками\n"
        "/roulette — русская рулетка\n"
        "/roast — прожарка случайного участника\n"
        "/achievements — твои достижения\n"
        "/top — топ чата\n\n"
        "Ещё отвечаю на вопросы про игры — упомяни меня через @.\n"
        "Попроси посоветовать кооп или одиночную PS5-игру — подберу с ценой.",
    )


__all__ = [
    "cmd_start",
    "cmd_help",
    "cmd_achievements",
    "cmd_top",
    "cmd_dnd_pvp",
    "cmd_dnd_coop",
    "cmd_dnd_heist",
    "handle_dnd_callback",
    "DND_CALLBACK_PATTERN",
    "cmd_duel",
    "handle_duel_callback",
    "DUEL_CALLBACK_PATTERN",
]
