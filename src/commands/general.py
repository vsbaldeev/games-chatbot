"""General bot commands — /start and /help."""

from telegram import Update
from telegram.ext import ContextTypes


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет. Я здесь. Спрашивайте про игры — если, конечно, есть что спросить."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет. Я здесь.\n\n"
        "Чтобы всё работало правильно:\n\n"
        "• Напишите что-нибудь в чат — каждый участник регистрируется, когда впервые пишет сообщение, реагирует или отправляет медиа. Только зарегистрированные попадают в дуэли.\n"
        "Что умею:\n"
        "/duel — эмодзи-дуэль между двумя участниками\n\n"
        "Ещё отвечаю на вопросы про игры — упомяни меня через @.\n"
        "Попроси посоветовать кооп или одиночную PS5-игру — подберу с ценой.",
    )
