"""Scheduled job: send one fresh meme to every known chat, once per day."""

import asyncio

from telegram.ext import ContextTypes

from src import achievements, config, log
from src.memes.fetcher import download_image, get_meme
from src.store import unified_messages

logger = log.get_logger(__name__)


async def send_meme_to_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Fetch one unseen meme and post it to the chat, recording it in history.

    Args:
        context: Telegram context used to send the photo.
        chat_id: The chat that should receive the daily meme.
    """
    result = await get_meme(chat_id)
    if result is None:
        logger.info("No unseen meme available for chat %s", chat_id)
        return
    image_url, caption = result
    image = await download_image(image_url)
    if image is None:
        return
    try:
        sent = await context.bot.send_photo(chat_id=chat_id, photo=image, caption=caption or None)
        file_id = sent.photo[-1].file_id if sent.photo else None
        await unified_messages.insert(
            chat_id=chat_id,
            message_id=sent.message_id,
            user_id=context.bot.id,
            username=config.BOT_USERNAME,
            content=unified_messages.format_photo_content(caption),
            media_type="photo",
            reply_to_msg_id=None,
            file_id=file_id,
        )
    except Exception as error:
        logger.warning("Daily meme failed for chat %s: %s", chat_id, error)


async def daily_meme_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a daily meme to every chat that has registered members."""
    chat_ids = await achievements.get_all_chat_ids()
    await asyncio.gather(
        *[send_meme_to_chat(context, chat_id) for chat_id in chat_ids],
        return_exceptions=True,
    )
