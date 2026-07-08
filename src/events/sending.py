"""Send-and-persist helper for the bot's out-of-pipeline messages.

Messages sent outside the pipeline (canned error replies, command outputs)
must still land in ``unified_messages`` — otherwise they are invisible to
recent history and future reply chains resolve them only via the
``replied_to_fallback``.
"""

from telegram import ReplyParameters

from src import config, log
from src.store import unified_messages

logger = log.get_logger(__name__)


async def send_and_store(bot, chat_id: int, text: str, *, reply_to: int | None = None):
    """Send a text message as the bot and persist it to ``unified_messages``.

    Args:
        bot: The ``telegram.Bot`` instance to send with.
        chat_id: Destination chat.
        text: Message text to send.
        reply_to: Message id to anchor the send to, or None for an
            un-anchored message. A deleted anchor degrades to un-anchored
            via ``allow_sending_without_reply``.

    Returns:
        The sent ``telegram.Message``.
    """
    reply_parameters = None
    if reply_to is not None:
        reply_parameters = ReplyParameters(
            message_id=reply_to, allow_sending_without_reply=True
        )
    sent = await bot.send_message(
        chat_id=chat_id, text=text, reply_parameters=reply_parameters
    )
    try:
        await unified_messages.insert(
            chat_id=chat_id,
            message_id=sent.message_id,
            user_id=config.BOT_ID,
            username=config.BOT_USERNAME,
            content=text,
            media_type="text",
            reply_to_msg_id=reply_to,
        )
    except Exception as err:
        logger.warning("Failed to store sent message %s: %s", sent.message_id, err)
    return sent
