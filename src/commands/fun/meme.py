"""Handler for the /meme command."""

from telegram import Update
from telegram.ext import ContextTypes

from src import config, log
from src.memes.fetcher import get_meme
from src.store import unified_messages

logger = log.get_logger(__name__)


async def cmd_meme(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    chat_id = update.effective_chat.id
    await update.message.chat.send_action("upload_photo")
    result = await get_meme(chat_id)
    if result is None:
        await update.message.reply_text("Мемы закончились — загляни попозже.")
        return
    url, caption = result
    try:
        sent = await update.message.reply_photo(url, caption=caption or None)
        file_id = sent.photo[-1].file_id if sent.photo else None
        await unified_messages.insert(
            chat_id=chat_id,
            message_id=sent.message_id,
            user_id=context.bot.id,
            username=config.BOT_USERNAME,
            content=unified_messages.format_photo_content(caption),
            media_type="photo",
            reply_to_msg_id=update.message.message_id,
            file_id=file_id,
        )
    except Exception as error:
        logger.error("Failed to send meme %s in chat %s: %s", url, chat_id, error)
        await update.message.reply_text("Не смог отправить мем — попробуй ещё раз.")
