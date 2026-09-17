"""Reply-to-bot thread tracking.

Lives outside src.events deliberately: this module is called from
src.pipeline.router, and src.pipeline never imports src.events (the
dependency runs the other way — src.events imports src.pipeline.graph).

unified_messages.get_chain already walks the reply_to_msg_id chain from a
message up to its root, oldest-first, capped at CHAIN_DEPTH_LIMIT hops — the
same helper the context builder uses for reply-chain assembly, reused here
rather than re-querying.
"""

from src import log
from src.feedback.classifier import classify_correction
from src.store import message_feedback, unified_messages

logger = log.get_logger(__name__)


async def track_reply(*, chat_id: int, message_id: int, bot_id: int) -> None:
    """Credit every bot message in the reply chain leading up to `message_id`.

    A reply chain can pass through more than one bot message (the member
    answers the bot, the bot answers back, the member replies again) — every
    bot ancestor is credited, each at its own hop distance below the
    triggering message, so a long back-and-forth counts toward every turn
    it actually continues, not just the newest one.

    Args:
        chat_id: Chat the reply was posted in.
        message_id: The newly stored reply message's id.
        bot_id: The bot's own user id.
    """
    try:
        chain = await unified_messages.get_chain(chat_id=chat_id, message_id=message_id)
    except Exception as err:
        logger.warning("Failed to load reply chain for message %s: %s", message_id, err)
        return
    # chain is oldest-first and includes message_id itself as the last entry;
    # depth is measured from the triggering message, so the last entry (the
    # reply) is excluded from the ancestor walk below, but kept as `reply`
    # for the depth-1 classification — get_chain already fetched its row, so
    # classifying it costs no second query.
    reply = chain[-1]
    ancestors = chain[:-1]
    try:
        for depth, ancestor in enumerate(reversed(ancestors), start=1):
            if ancestor["user_id"] != bot_id:
                continue
            await message_feedback.add_reply(chat_id=chat_id, message_id=ancestor["message_id"], depth=depth)
            if depth == 1:
                await _classify_direct_reply(chat_id=chat_id, bot_message=ancestor, reply=reply)
    except Exception as err:
        logger.warning("Failed to credit reply for message %s: %s", message_id, err)


async def _classify_direct_reply(*, chat_id: int, bot_message: dict, reply: dict) -> None:
    """Classify a depth-1 reply to a bot message as a correction, if it is text.

    Only the direct reply is classified — deeper hops in the same thread are
    still credited toward thread_depth/replies by track_reply's caller, but
    classifying every one of them would multiply LLM calls for marginal
    signal over just checking the reply that actually answers the bot.
    """
    if reply["media_type"] != "text" or not reply["content"]:
        return
    is_correction = await classify_correction(bot_message["content"], reply["content"])
    if is_correction:
        await message_feedback.add_correction(chat_id=chat_id, message_id=bot_message["message_id"])
