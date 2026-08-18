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


def build_reply_parameters(reply_to: int | None) -> ReplyParameters | None:
    """Build reply parameters for an anchored send, or None for an un-anchored one.

    Args:
        reply_to: Message id to anchor the send to, or None for an un-anchored message.

    Returns:
        ReplyParameters with allow_sending_without_reply=True, or None if reply_to is None.
    """
    if reply_to is None:
        return None
    return ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)


async def send_and_store(
    bot, chat_id: int, text: str, *, reply_to: int | None = None, is_broadcast: bool = False
):
    """Send a text message as the bot and persist it to ``unified_messages``.

    Args:
        bot: The ``telegram.Bot`` instance to send with.
        chat_id: Destination chat.
        text: Message text to send.
        reply_to: Message id to anchor the send to, or None for an
            un-anchored message. A deleted anchor degrades to un-anchored
            via ``allow_sending_without_reply``.
        is_broadcast: True when this is a group-wide announcement about the
            members (weekly roles, group profile). Persisted so the router's
            addressing gate can treat replies to it as chat among members
            rather than as messages to the bot.

    Returns:
        The sent ``telegram.Message``.
    """
    reply_parameters = build_reply_parameters(reply_to)
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
            is_broadcast=is_broadcast,
        )
    except Exception as err:
        logger.warning("Failed to store sent message %s: %s", sent.message_id, err)
    return sent
