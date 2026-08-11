"""Sends a vetted meme to a chat and records it in message history.

The single place a meme reaches Telegram. Both callers — the daily job and the
on-request path in the events layer — differ only in whether the image is
anchored to a requesting message, so they share this rather than each carrying
their own copy of the send-and-record sequence.

Captions are never attached: the scraped caption belongs to the source channel,
and a caption the vision gate never approved is exactly what once made the bot
look like it was asking readers for donations. Storing the bare photo
placeholder also leaves the row for the existing photo-description enricher to
fill in with a real description of the image.
"""

from telegram import ReplyParameters

from src import config, log
from src.memes.fetcher import get_meme
from src.store import unified_messages

logger = log.get_logger(__name__)


def build_reply_parameters(reply_to: int | None) -> ReplyParameters | None:
    """Build reply parameters anchoring a send to a message, if there is one.

    Args:
        reply_to: Message id to anchor to, or None for an un-anchored send.

    Returns:
        ``ReplyParameters`` tolerating a deleted anchor, or None.
    """
    if reply_to is None:
        return None
    return ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)


async def send_meme(bot, chat_id: int, *, reply_to: int | None = None) -> bool:
    """Fetch one vetted meme and post it to the chat.

    Args:
        bot: The ``telegram.Bot`` instance to send with.
        chat_id: Destination chat.
        reply_to: Message id to anchor the photo to, or None for an
            un-anchored send. A deleted anchor degrades to un-anchored via
            ``allow_sending_without_reply``.

    Returns:
        True when a meme was sent, False when none was available — the pool is
        exhausted for this chat, every candidate was rejected, or the
        fail-closed vision gate had no verdict. Send failures also return
        False; they are logged, never raised.
    """
    image = await get_meme(chat_id)
    if image is None:
        logger.info("No vetted meme available for chat %s", chat_id)
        return False
    try:
        sent = await bot.send_photo(
            chat_id=chat_id, photo=image,
            reply_parameters=build_reply_parameters(reply_to),
        )
        await unified_messages.insert(
            chat_id=chat_id,
            message_id=sent.message_id,
            user_id=bot.id,
            username=config.BOT_USERNAME,
            content=unified_messages.format_photo_content(None),
            media_type="photo",
            reply_to_msg_id=reply_to,
            file_id=sent.photo[-1].file_id if sent.photo else None,
        )
        return True
    except Exception as error:
        logger.warning("Failed to send meme to chat %s: %s", chat_id, error)
        return False
