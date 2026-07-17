"""Chat-requested selfie generation — «Жора, сфоткай себя».

Runs as a fire-and-forget background task after the pipeline delivered the
in-character «ща, сфоткаю» acknowledgement: an LLM turns the member's Russian
request into one English scene line, ``generate_best_photo`` renders and
judges the candidates, and the photo arrives minutes later as a reply to the
requesting message. Any failure degrades to a canned in-character excuse —
the task never raises.

One global generation slot guards the whole module: the imagegen service has
a single worker shared with scheduled life posts, so a second chat request
while one is rendering gets an «уже фоткаю» acknowledgement (the filter peeks
``is_generation_in_flight``) and no second job. The peek happens at
classification time and the acquire here, so two overlapping pipelines can
both ack while only one generates; the loser logs and exits — a rare,
low-stakes race (one user gets an ack with no photo).
"""

import random

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import ReplyParameters

from src import config, log
from src.agent.middleware import ainvoke_with_backoff, strip_thinking
from src.config.prompts import SELFIE_SCENE_SYSTEM
from src.events.sending import send_and_store
from src.life.poster import generate_best_photo
from src.store import unified_messages

logger = log.get_logger(__name__)

# Short in-character captions under a delivered selfie — deterministic pool,
# no LLM call, mirroring the canned-reply convention of the filter node.
SELFIE_CAPTIONS = [
    "Держи.",
    "Во, снял.",
    "Как заказывали.",
    "На, любуйся.",
]

# In-character excuses when scene writing or generation failed — honest
# degradation instead of a silent hang after the «ща сфоткаю» ack.
SELFIE_FAILED_REPLIES = [
    "Камера навернулась, фотки не будет.",
    "Телефон сел. В другой раз.",
    "Фотка не отправляется — интернет у меня деревенский, сам знаешь.",
    "Пока фоткал, батарея сдохла. Потом покажу.",
]

# Single global slot for the whole module: the imagegen service has one
# worker, so at most one chat-requested selfie renders at a time.
generation_in_flight = False


def is_generation_in_flight() -> bool:
    """Peek whether a chat-requested selfie is currently being generated.

    Synchronous read used by the filter node to ack «уже фоткаю» instead of
    promising a second photo.

    Returns:
        True while :func:`deliver_selfie` holds the generation slot.
    """
    return generation_in_flight


def build_scene_input(request_text: str, current_activity: str | None) -> str:
    """Assemble the human message for the selfie scene writer.

    Args:
        request_text: The member's raw Russian photo request.
        current_activity: Жора's current-activity phrase from the newest life
            post, or None when unknown.

    Returns:
        The bare request, with the current activity appended on its own
        labelled line when available (the prompt's optional activity line).
    """
    if current_activity:
        return f"{request_text}\nThe man is currently busy with: {current_activity}"
    return request_text


async def write_selfie_scene(request_text: str, current_activity: str | None) -> str | None:
    """Turn the member's photo request into one English scene description.

    Args:
        request_text: The member's raw Russian photo request.
        current_activity: Жора's current-activity phrase, or None.

    Returns:
        The trimmed scene line, or None on any LLM error or empty output —
        the caller then degrades to a canned excuse.
    """
    llm = ChatGroq(
        model=config.SELFIE_SCENE_MODEL, api_key=config.GROQ_API_KEY,
        temperature=0.4, max_tokens=config.SELFIE_SCENE_MAX_TOKENS, max_retries=0,
    )
    try:
        result = await ainvoke_with_backoff(llm, [
            SystemMessage(content=SELFIE_SCENE_SYSTEM),
            HumanMessage(content=build_scene_input(request_text, current_activity)),
        ])
    except Exception as error:
        logger.warning("Selfie scene writer failed: %s", error)
        return None
    scene = strip_thinking(result.content or "").strip()
    if not scene:
        logger.warning("Selfie scene writer returned empty output")
        return None
    return scene


async def send_excuse(bot, chat_id: int, reply_to_msg_id: int) -> None:
    """Reply with a canned in-character excuse instead of the promised photo.

    Args:
        bot: Telegram Bot instance to send with.
        chat_id: Chat the request came from.
        reply_to_msg_id: The requesting message to anchor the excuse to.
    """
    await send_and_store(
        bot, chat_id, random.choice(SELFIE_FAILED_REPLIES), reply_to=reply_to_msg_id
    )


async def send_and_record_photo(bot, chat_id: int, reply_to_msg_id: int, photo_png: bytes) -> None:
    """Send the generated selfie and persist it to ``unified_messages``.

    The row stores placeholder photo content plus the Telegram ``file_id``
    (mirroring ``poster.record_sent_message``), which plugs the selfie into
    the existing lazy vision-description path when a member replies to it.

    Args:
        bot: Telegram Bot instance to send with.
        chat_id: Chat the request came from.
        reply_to_msg_id: The requesting message the photo replies to.
        photo_png: PNG bytes of the chosen candidate.
    """
    caption = random.choice(SELFIE_CAPTIONS)
    sent = await bot.send_photo(
        chat_id=chat_id,
        photo=photo_png,
        caption=caption,
        reply_parameters=ReplyParameters(
            message_id=reply_to_msg_id, allow_sending_without_reply=True
        ),
    )
    await unified_messages.insert(
        chat_id=chat_id,
        message_id=sent.message_id,
        user_id=config.BOT_ID,
        username=config.BOT_USERNAME,
        content=unified_messages.format_photo_content(caption),
        media_type="photo",
        reply_to_msg_id=reply_to_msg_id,
        file_id=sent.photo[-1].file_id if sent.photo else None,
    )


async def deliver_selfie(
    *, bot, chat_id: int, reply_to_msg_id: int,
    request_text: str, current_activity: str | None,
) -> None:
    """Generate and deliver one chat-requested selfie; never raises.

    Background-task body launched after the acknowledgement was delivered:
    writes the scene, renders the best candidate, sends it as a reply to the
    request and records it. Any failure degrades to a canned in-character
    excuse; losing the generation slot (peek/acquire race) exits silently —
    one photo is already coming.

    Args:
        bot: Telegram Bot instance to send with.
        chat_id: Chat the request came from.
        reply_to_msg_id: The requesting message id to anchor replies to.
        request_text: The member's raw Russian photo request.
        current_activity: Жора's current-activity phrase, or None.
    """
    global generation_in_flight
    if generation_in_flight:
        logger.info("Selfie already in flight — dropping request in chat %s", chat_id)
        return
    generation_in_flight = True
    try:
        scene = await write_selfie_scene(request_text, current_activity)
        if scene is None:
            await send_excuse(bot, chat_id, reply_to_msg_id)
            return
        logger.info("Selfie scene for chat %s: %s", chat_id, scene)
        photo_png = await generate_best_photo(scene)
        if photo_png is None:
            await send_excuse(bot, chat_id, reply_to_msg_id)
            return
        await send_and_record_photo(bot, chat_id, reply_to_msg_id, photo_png)
    except Exception as error:
        logger.warning("Selfie delivery failed for chat %s: %s", chat_id, error)
    finally:
        generation_in_flight = False
