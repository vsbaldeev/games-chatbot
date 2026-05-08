"""Handler for the /meme command."""

from telegram import Update
from telegram.ext import ContextTypes

from src import log
from src.memes.fetcher import get_meme

logger = log.get_logger(__name__)


async def cmd_meme(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await update.message.chat.send_action("upload_photo")
    result = await get_meme(chat_id)
    if result is None:
        await update.message.reply_text("Мемы закончились — Reddit нас подвёл.")
        return
    url, caption = result
    try:
        await update.message.reply_photo(url, caption=caption or None)
    except Exception as error:
        logger.error("Failed to send meme %s in chat %s: %s", url, chat_id, error)
        await update.message.reply_text("Не смог отправить мем — попробуй ещё раз.")
