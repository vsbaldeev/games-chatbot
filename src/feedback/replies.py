"""Reply-to-bot counting.

Lives outside src.events deliberately: this module is called from
src.pipeline.router, and src.pipeline never imports src.events (the
dependency runs the other way — src.events imports src.pipeline.graph).

track_reply finds the nearest bot message in the reply chain and hands the
reply to a background task, so a labelling call can never slow or break the
reply path. The task labels the reply (one OpenRouter call for text, a rule
for everything else) and bumps one counter on that bot message's
bot_message_feedback row. Only the nearest bot message is credited, so a long
back-and-forth does not inflate every earlier turn.
"""

import asyncio

from src import log
from src.feedback.reply_labeller import LabellingCallFailed, label_reply
from src.store import message_feedback, unified_messages

logger = log.get_logger(__name__)

# Triggers whose replies are not measured at all: Reels and long-form YouTube
# summaries are not a signal the operator reads, so labelling replies to them
# would spend model calls for nothing.
EXCLUDED_TRIGGERS: frozenset[str] = frozenset({"social_link"})

LABEL_KINDS: dict[str, str] = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
}

# Strong references to in-flight counting tasks: the event loop only keeps a
# weak one, so an unreferenced task can be garbage-collected mid-call.
BACKGROUND_TASKS: set[asyncio.Task] = set()


def nearest_bot_ancestor(chain: list[dict], bot_id: int) -> dict | None:
    """Return the newest bot message in an oldest-first chain, or None."""
    for message in reversed(chain):
        if message["user_id"] == bot_id:
            return message
    return None


async def classify_reply(*, bot_message: dict, reply: dict, source: str) -> str:
    """Decide which reply counter this reply bumps (a key of REPLY_KIND_COLUMNS)."""
    if reply["media_type"] != "text":
        return "media"
    if not reply["content"].strip():
        return "unlabelled"
    try:
        label = await label_reply(bot_message=bot_message["content"], reply=reply["content"], source=source)
    except LabellingCallFailed as error:
        logger.warning("Reply labelling call failed for message %s: %s", reply["message_id"], error)
        return "unlabelled"
    return LABEL_KINDS.get(label, "unlabelled")


async def count_reply(*, chat_id: int, bot_message: dict, reply: dict) -> None:
    """Label one reply and bump the matching counter. Never raises."""
    bot_message_id = bot_message["message_id"]
    try:
        origin = await message_feedback.get_origin(chat_id=chat_id, message_id=bot_message_id)
        if origin is None:
            return
        source, trigger = origin
        if trigger in EXCLUDED_TRIGGERS:
            return
        kind = await classify_reply(bot_message=bot_message, reply=reply, source=source)
        await message_feedback.add_reply(chat_id=chat_id, message_id=bot_message_id, kind=kind)
    except Exception as error:
        logger.warning("Failed to count reply %s on bot message %s: %s", reply["message_id"], bot_message_id, error)


async def track_reply(*, chat_id: int, message_id: int, bot_id: int) -> None:
    """Start counting a reply against the nearest bot message in its chain.

    Args:
        chat_id: Chat the reply was posted in.
        message_id: The newly stored reply message's id.
        bot_id: The bot's own user id.
    """
    try:
        chain = await unified_messages.get_chain(chat_id=chat_id, message_id=message_id)
    except Exception as error:
        logger.warning("Failed to load reply chain for message %s: %s", message_id, error)
        return
    if not chain:
        return
    reply = chain[-1]
    bot_message = nearest_bot_ancestor(chain[:-1], bot_id)
    if bot_message is None:
        return
    task = asyncio.create_task(count_reply(chat_id=chat_id, bot_message=bot_message, reply=reply))
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)
