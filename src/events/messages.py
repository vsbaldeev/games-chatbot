"""Message handlers — text, voice, photo, sticker, video."""

import datetime
import re
import time

from src import log

from telegram import ReplyParameters, Update
from telegram.ext import ContextTypes

import asyncio

from src import achievements, config
from src.agent import (
    worker_agent,
    response_agent,
    ContextLengthError,
    DailyLimitError,
    RateLimitError,
    normalize_homoglyphs,
)
from src.pipeline import canonical
from src.pipeline.graph import build_pipeline
from src.events.members import get_username
from src.pipeline.ingester import transcribe_voice
from src.pipeline.memory_writer import MIN_PASSIVE_LENGTH, extract_and_save
from src.events.sending import send_and_store
from src.events.voice_reply import try_send_voice_reply
from src.pipeline.router import is_explicitly_addressed
from src.store import unified_messages
from src.utils.ttl_gate import TtlGate

PIPELINE = build_pipeline(worker_agent, response_agent)

MOSCOW_TZ = datetime.timezone(datetime.timedelta(hours=3))

LIMIT_NOTICE_COOLDOWN_SECONDS = 30 * 60
quota_notice_gate = TtlGate(LIMIT_NOTICE_COOLDOWN_SECONDS)

DAILY_LIMIT_NOTICE = (
    "📵 Суточный лимит токенов Groq исчерпан. Бот ушёл спать до завтра. "
    "Статья на Луркоморье: «Бесплатный тариф — он такой»."
)
RATE_LIMIT_NOTICE = (
    "⏳ Groq не завезли лимитов. Бот временно на перекуре — слишком много запросов. "
    "Попробуйте через минуту, анончики."
)
GENERIC_FAILURE_NOTICE = (
    "Что-то сломалось. Скорее всего, Groq опять тупит. Попробуй позже."
)

# Incoming media types answered in kind with a voice note.
VOICE_REPLY_TRIGGER_MEDIA_TYPES = ("voice", "video_note")


def is_night_message(update: Update) -> bool:
    if not update.message or not update.message.date:
        return False
    return 0 <= update.message.date.astimezone(MOSCOW_TZ).hour < 5


def is_reply_to_game_message(update: Update) -> bool:
    reply = update.message.reply_to_message
    if not reply:
        return False
    text = reply.text or ""
    return text.startswith(("⚔️", "🎩", "🔫", "💀"))


from src.pipeline.response_node import strip_markdown
from src.pipeline.state import BotState, IncomingMessage
from src.store import thread_history

REPLY_PLACEHOLDERS = {
    "voice": unified_messages.VOICE_PLACEHOLDER,
    "video_note": unified_messages.VIDEO_NOTE_PLACEHOLDER,
    "video": unified_messages.VIDEO_PLACEHOLDER,
    "sticker": unified_messages.STICKER_PLACEHOLDER,
    "animation": unified_messages.ANIMATION_PLACEHOLDER,
    "audio": unified_messages.AUDIO_PLACEHOLDER,
}


async def derive_thread_id(chat_id: int, reply_to_msg_id: int | None) -> str:
    """Return a thread_id scoped to the reply chain root, or chat_id for flat messages.

    Replies whose chain is unknown to the store (the replied-to message was
    never persisted) get their own ``{chat_id}_{reply_to_msg_id}`` scope
    instead of inheriting the stale flat bucket.
    """
    if reply_to_msg_id is None:
        return str(chat_id)
    chain = await unified_messages.get_chain(chat_id=chat_id, message_id=reply_to_msg_id)
    if not chain:
        return thread_history.thread_id_for_root(chat_id, reply_to_msg_id)
    root = chain[0]  # get_chain returns oldest-first
    return thread_history.thread_id_for_root(chat_id, root["message_id"])


def derive_reply_media_type(reply) -> str:
    """Classify a replied-to Telegram message into the store's media types.

    Args:
        reply: A ``telegram.Message`` object taken from ``reply_to_message``.

    Returns:
        One of the media-type strings used in ``unified_messages`` rows;
        ``"text"`` when no known media attachment is present.
    """
    for media_type in ("voice", "video_note", "video", "photo", "sticker", "animation", "audio"):
        if getattr(reply, media_type, None):
            return media_type
    return "text"


def build_replied_to_fallback(reply) -> dict | None:
    """Synthesize a ``unified_messages``-row-shaped dict from a reply object.

    Telegram includes the complete replied-to message in every reply update.
    When the store has no row for it (other bots' posts, command outputs,
    expired or missed messages), this fallback keeps reply context working.
    It is read-side only — never inserted into the store.

    Args:
        reply: ``msg.reply_to_message``, or ``None`` for non-replies.

    Returns:
        A dict with the same keys consumers expect from a stored row, or
        ``None`` when there is no reply or no identifiable sender.
    """
    if reply is None:
        return None
    sender = getattr(reply, "from_user", None)
    if sender is None:
        return None
    media_type = derive_reply_media_type(reply)
    if media_type == "photo":
        content = unified_messages.format_photo_content(reply.caption)
    elif media_type == "text":
        content = reply.text or reply.caption or ""
    else:
        content = REPLY_PLACEHOLDERS[media_type]
    nested_reply = getattr(reply, "reply_to_message", None)
    return {
        "message_id": reply.message_id,
        "user_id": sender.id,
        "username": sender.username or sender.first_name or f"user_{sender.id}",
        "content": content,
        "media_type": media_type,
        "reply_to_msg_id": nested_reply.message_id if nested_reply else None,
        "file_id": None,
    }

logger = log.get_logger(__name__)

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027FF\U0001FA00-\U0001FAFF]",
    re.UNICODE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def build_pipeline_state(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    media_type: str,
    file_id: str | None,
) -> BotState:
    msg = update.message
    user = update.effective_user
    chat = update.effective_chat
    username = get_username(update)

    reply_to_msg_id: int | None = None
    if msg.reply_to_message:
        reply_to_msg_id = msg.reply_to_message.message_id

    incoming: IncomingMessage = {
        "update": update,
        "chat_id": chat.id,
        "user_id": user.id,
        "username": username,
        "raw_text": msg.text or msg.caption or None,
        "processed_text": None,
        "media_type": media_type,
        "message_id": msg.message_id,
        "reply_to_msg_id": reply_to_msg_id,
        "file_id": file_id,
        "is_forwarded": msg.forward_origin is not None,
        "media_group_id": msg.media_group_id,
        "replied_to_fallback": build_replied_to_fallback(msg.reply_to_message),
    }
    return {
        "incoming": incoming,
        "should_respond": False,
        "response_trigger": "random",
        "blocked": False,
        "context": None,
        "response": None,
        "context_types": context,
    }


async def deliver_response(final_state: BotState, msg, clean: str) -> tuple[int, int | None, str]:
    """Send the pipeline response to Telegram.

    Normal responses reply to the triggering message; when the trigger was a
    voice message or video note the reply goes out in kind as a voice note,
    degrading to text on any synthesis failure. Autonomous jokes
    (``response_trigger == "humor"``) instead anchor to the message the comedian
    cited, or go out un-anchored when there is no validated target; a cited
    message deleted in the meantime degrades to un-anchored via
    ``allow_sending_without_reply``.

    Args:
        final_state: Pipeline state after the graph run.
        msg: The triggering ``telegram.Message``.
        clean: Markdown-stripped response text.

    Returns:
        Tuple of the sent message id, the message id the response is anchored
        to (None for an un-anchored joke), and the sent media type
        ("text" or "voice").
    """
    notification_msg = final_state.get("search_notification_msg")
    if notification_msg:
        await notification_msg.edit_text(clean)
        return notification_msg.message_id, msg.message_id, "text"
    await msg.chat.send_action("typing")
    if final_state.get("response_trigger") == "humor":
        target = final_state.get("humor_reply_to_msg_id")
        reply_parameters = None
        if target is not None:
            reply_parameters = ReplyParameters(message_id=target, allow_sending_without_reply=True)
        sent = await msg.get_bot().send_message(
            chat_id=msg.chat_id, text=clean, reply_parameters=reply_parameters
        )
        return sent.message_id, target, "text"
    if final_state["incoming"]["media_type"] in VOICE_REPLY_TRIGGER_MEDIA_TYPES:
        voice_message = await try_send_voice_reply(msg, clean)
        if voice_message is not None:
            return voice_message.message_id, msg.message_id, "voice"
    sent = await msg.reply_text(clean)
    return sent.message_id, msg.message_id, "text"


async def send_limit_notice(msg, chat_id: int, notice_text: str) -> None:
    """Send a quota/rate-limit notice, throttled to one per chat per cooldown.

    The first notice within ``LIMIT_NOTICE_COOLDOWN_SECONDS`` goes out as a
    full text reply; while the cooldown is active the user gets a 😴 reaction
    instead, so an addressed user is still acknowledged without the bot
    re-posting the same wall of text on every mention.

    Args:
        msg: The triggering ``telegram.Message`` to reply or react to.
        chat_id: Chat the notice belongs to (cooldown is per chat).
        notice_text: The full notice to send when outside the cooldown.
    """
    if not quota_notice_gate.seen(chat_id):
        await send_and_store(msg.get_bot(), chat_id, notice_text, reply_to=msg.message_id)
        return
    try:
        await msg.set_reaction("😴")
    except Exception as err:
        logger.warning("Failed to react with cooldown emoji in chat %s: %s", chat_id, err)


def build_context_length_notice(msg) -> str:
    """Pick the context-overflow advice matching how the message arrived.

    Args:
        msg: The triggering ``telegram.Message``.

    Returns:
        Chain-specific advice when the message is part of a reply chain,
        plain "message too long" advice otherwise.
    """
    if msg.reply_to_message:
        return (
            "Цепочка ответов слишком длинная — не влезает в контекст модели. "
            "Начни новое сообщение вместо ответа на старое."
        )
    return (
        "Сообщение слишком длинное — не влезает в контекст модели. "
        "Сократи и попробуй ещё раз."
    )


async def deliver_and_record(final_state, msg, bot_id: int, response_text: str) -> None:
    """Deliver the pipeline response and store the sent message.

    Args:
        final_state: Final pipeline state (drives the delivery mode).
        msg: The triggering ``telegram.Message``.
        bot_id: The bot's own user id, recorded as the message author.
        response_text: Raw response text produced by the pipeline.
    """
    clean = normalize_homoglyphs(strip_markdown(response_text))
    sent_id, anchored_to, sent_media_type = await deliver_response(final_state, msg, clean)
    await unified_messages.insert(
        chat_id=msg.chat_id,
        message_id=sent_id,
        user_id=bot_id,
        username=config.BOT_USERNAME,
        content=clean,
        media_type=sent_media_type,
        reply_to_msg_id=anchored_to,
    )


async def notify_pipeline_failure(error: Exception, msg, chat_id: int, addressed: bool) -> str:
    """Log a pipeline failure, notify the chat when addressed, name the kind.

    Args:
        error: The exception raised by the pipeline.
        msg: The triggering ``telegram.Message``.
        chat_id: Chat the failure happened in.
        addressed: True when the user explicitly addressed the bot; only then
            is a notice posted to the chat.

    Returns:
        Short error kind for the canonical log line, e.g. ``"DailyLimit"``.
    """
    if isinstance(error, DailyLimitError):
        logger.warning("Daily token quota exhausted for chat %s", chat_id)
        if addressed:
            await send_limit_notice(msg, chat_id, DAILY_LIMIT_NOTICE)
        return "DailyLimit"
    if isinstance(error, ContextLengthError):
        logger.warning("Context length exceeded for chat %s", chat_id)
        if addressed:
            await send_and_store(
                msg.get_bot(), chat_id, build_context_length_notice(msg),
                reply_to=msg.message_id,
            )
        return "ContextLength"
    if isinstance(error, RateLimitError):
        logger.warning("Rate limit reached for chat %s", chat_id)
        if addressed:
            await send_limit_notice(msg, chat_id, RATE_LIMIT_NOTICE)
        return "RateLimit"
    logger.error("Pipeline error for chat %s: %s", chat_id, error, exc_info=True)
    if addressed:
        await send_and_store(
            msg.get_bot(), chat_id, GENERIC_FAILURE_NOTICE,
            reply_to=msg.message_id,
        )
    return "Exception"


async def run_pipeline(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    media_type: str,
    file_id: str | None = None,
) -> bool:
    """Run the LangGraph pipeline and return True if the bot sent a response.

    Binds a per-update correlation id for log grouping and emits one
    canonical INFO log line summarizing the run outcome. Errors are reported
    in chat only when the user explicitly addressed the bot (mention or
    reply-to-bot); autonomous paths (random media rolls, overheard insult
    checks) fail silently with a log-only warning.
    """
    log.bind_correlation_id(str(update.update_id)[-6:])
    started_at = time.monotonic()
    msg = update.message
    chat = update.effective_chat
    reply_to_msg_id = msg.reply_to_message.message_id if msg.reply_to_message else None
    thread_id = await derive_thread_id(chat.id, reply_to_msg_id)
    initial_state = build_pipeline_state(update, context, media_type, file_id)
    initial_state["thread_id"] = thread_id
    initial_state["is_flat_thread"] = reply_to_msg_id is None
    addressed = is_explicitly_addressed(msg, config.BOT_USERNAME, config.BOT_ID)
    final_state = initial_state

    try:
        final_state = await PIPELINE.ainvoke(initial_state)
        response = final_state.get("response") or ""
        if response.strip():
            await deliver_and_record(final_state, msg, context.bot.id, response)
            action = "joked" if final_state.get("response_trigger") == "humor" else "replied"
            canonical.emit(final_state, action, time.monotonic() - started_at)
            return True
    except Exception as error:
        error_kind = await notify_pipeline_failure(error, msg, chat.id, addressed)
        canonical.emit(final_state, f"error:{error_kind}", time.monotonic() - started_at)
        return False
    canonical.emit(final_state, "ignored", time.monotonic() - started_at)
    return False


async def passive_voice_extract(
    *, file_id: str, media_type: str, bot,
    chat_id: int, user_id: int, username: str,
) -> None:
    transcript = await transcribe_voice(file_id, media_type, bot)
    if transcript and len(transcript.strip()) >= MIN_PASSIVE_LENGTH:
        await extract_and_save(
            chat_id=chat_id, user_id=user_id, username=username,
            user_message=transcript, source_kind="voice",
        )


async def track_text_stats(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    chat_id: int,
    username: str,
    text: str,
) -> None:
    if is_night_message(update):
        await achievements.increment_stat(user_id, chat_id, username, "night_messages")
    if EMOJI_RE.search(text):
        await achievements.increment_stat(user_id, chat_id, username, "emoji_messages")
    if URL_RE.search(text):
        await achievements.increment_stat(user_id, chat_id, username, "link_messages")
    if update.message.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
    await achievements.update_max_stat(user_id, chat_id, username, "long_message_max", len(text))



async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    if update.effective_user and update.effective_user.is_bot:
        return

    text = update.message.text
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    await track_text_stats(update, context, user_id, chat_id, username, text)

    if is_reply_to_game_message(update):
        await store_game_reply(update, chat_id, user_id, username, text)
        return

    await run_pipeline(update, context, media_type="text")


async def store_game_reply(
    update: Update, chat_id: int, user_id: int, username: str, text: str
) -> None:
    """Persist a user reply to a game message that skips the pipeline.

    Game replies never trigger a response, but they must still land in
    ``unified_messages`` — otherwise they punch holes in recent history and
    in future reply chains.

    Args:
        update: The Telegram update carrying the reply.
        chat_id: Chat the reply was posted in.
        user_id: Author of the reply.
        username: Author's display username.
        text: Message text.
    """
    msg = update.message
    reply_to_msg_id = msg.reply_to_message.message_id if msg.reply_to_message else None
    try:
        await unified_messages.insert(
            chat_id=chat_id,
            message_id=msg.message_id,
            user_id=user_id,
            username=username,
            content=text,
            media_type="text",
            reply_to_msg_id=reply_to_msg_id,
            is_forwarded=msg.forward_origin is not None,
        )
    except Exception as err:
        logger.warning("Failed to store game reply %s: %s", msg.message_id, err)


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return
    if update.effective_user and update.effective_user.is_bot:
        return

    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if msg.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
        if msg.voice:
            await run_pipeline(update, context, media_type="voice", file_id=msg.voice.file_id)
        elif msg.video_note:
            await run_pipeline(update, context, media_type="video_note", file_id=msg.video_note.file_id)
        return

    if msg.voice:
        media_type = "voice"
        file_id = msg.voice.file_id
        await achievements.increment_stat(user_id, chat_id, username, "voice_messages")
        await achievements.update_max_stat(user_id, chat_id, username, "voice_max_duration", msg.voice.duration)
    elif msg.video_note:
        media_type = "video_note"
        file_id = msg.video_note.file_id
        await achievements.increment_stat(user_id, chat_id, username, "video_note_messages")
    else:
        return

    responded = await run_pipeline(update, context, media_type=media_type, file_id=file_id)
    if not responded:
        asyncio.create_task(passive_voice_extract(
            file_id=file_id, media_type=media_type, bot=context.bot,
            chat_id=chat_id, user_id=user_id, username=username,
        ))


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.photo:
        return
    if update.effective_user and update.effective_user.is_bot:
        return

    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    photo = msg.photo[-1]
    if msg.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
        await run_pipeline(update, context, media_type="photo", file_id=photo.file_id)
        return

    await achievements.increment_stat(user_id, chat_id, username, "photo_messages")
    await run_pipeline(update, context, media_type="photo", file_id=photo.file_id)


async def handle_sticker_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.sticker:
        return
    if update.effective_user and update.effective_user.is_bot:
        return
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    if msg.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
        await run_pipeline(update, context, media_type="sticker", file_id=msg.sticker.file_id)
        return
    await achievements.increment_stat(user_id, chat_id, username, "sticker_messages")
    await run_pipeline(update, context, media_type="sticker", file_id=msg.sticker.file_id)


async def handle_animation_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.animation:
        return
    if update.effective_user and update.effective_user.is_bot:
        return
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    if msg.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
        await run_pipeline(update, context, media_type="animation", file_id=msg.animation.file_id)
        return
    await achievements.increment_stat(user_id, chat_id, username, "animation_messages")
    await run_pipeline(update, context, media_type="animation", file_id=msg.animation.file_id)


async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.audio:
        return
    if update.effective_user and update.effective_user.is_bot:
        return
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    if msg.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
        await run_pipeline(update, context, media_type="audio", file_id=msg.audio.file_id)
        return
    await run_pipeline(update, context, media_type="audio", file_id=msg.audio.file_id)


async def handle_video_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.video:
        return
    if update.effective_user and update.effective_user.is_bot:
        return
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    if msg.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
        await run_pipeline(update, context, media_type="video", file_id=msg.video.file_id)
        return
    await achievements.increment_stat(user_id, chat_id, username, "video_messages")
    await run_pipeline(update, context, media_type="video", file_id=msg.video.file_id)
