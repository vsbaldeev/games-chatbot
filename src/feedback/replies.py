"""Reply-to-bot thread tracking.

Lives outside src.events deliberately: this module is called from
src.pipeline.router, and src.pipeline never imports src.events (the
dependency runs the other way — src.events imports src.pipeline.graph).

unified_messages.get_chain already walks the reply_to_msg_id chain from a
message up to its root, oldest-first, capped at CHAIN_DEPTH_LIMIT hops — the
same helper the context builder uses for reply-chain assembly, reused here
rather than re-querying.

Labelling is not done here. track_reply enqueues the reply into
bot_reply_labels and returns; src/jobs/reply_labels.py labels it within
fifteen minutes. That keeps every model call off the message-handling path,
so a labelling outage cannot slow or break a reply.
"""

from src import log
from src.store import message_feedback, reply_labels, unified_messages

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
    # for the enqueue call — get_chain already fetched its row, so enqueuing
    # it costs no second query.
    reply = chain[-1]
    ancestors = chain[:-1]
    try:
        enqueued = False
        for depth, ancestor in enumerate(reversed(ancestors), start=1):
            if ancestor["user_id"] != bot_id:
                continue
            await message_feedback.add_reply(chat_id=chat_id, message_id=ancestor["message_id"])
            if not enqueued:
                await reply_labels.enqueue(
                    chat_id=chat_id,
                    reply_message_id=reply["message_id"],
                    bot_message_id=ancestor["message_id"],
                    depth=depth,
                    media_type=reply["media_type"],
                )
                enqueued = True
    except Exception as err:
        logger.warning("Failed to credit reply for message %s: %s", message_id, err)
