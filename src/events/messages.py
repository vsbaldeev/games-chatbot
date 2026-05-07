"""Message handlers — text, voice, photo, sticker, video."""

import datetime
import re
from src import log

from telegram import Update
from telegram.ext import ContextTypes

from src import achievements, config
from src.agent import agent, DailyLimitError, RateLimitError
from src.achievements import notify_unlocks
from src.events.members import get_username

OFFENSE_RE = re.compile(
    r"(тупой|тупая|тупит|идиот|дебил|мудак|г[ао]вн[оа]|хуйн[яе]|нахуй|пиздец|"
    r"отстой|бесполезн|сломан|не работает|глупый|глупая|дерьм[оа]|придур|долбо|"
    r"ёбан|еба[нл]|заткн|иди нах|иди в|stupid|useless|broken|dumb|trash|"
    r"garbage|sucks|piece of shit|fuck)",
    re.IGNORECASE | re.UNICODE,
)


TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


MOSCOW_TZ = datetime.timezone(datetime.timedelta(hours=3))


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


def strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_(.+?)_", r"\1", text, flags=re.DOTALL)
    lines = [line for line in text.splitlines() if not TABLE_SEP_RE.match(line)]
    return "\n".join(lines)
from src.pipeline.state import BotState, IncomingMessage
from src.commands.fun.roast import generate_roast_text

logger = log.get_logger(__name__)

WHISPER_MODEL = "whisper-large-v3"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027FF\U0001FA00-\U0001FAFF]",
    re.UNICODE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# {chat_id: {user_id: count}} — tracks consecutive offensive replies toward the bot
offense_reply_counts: dict[int, dict[int, int]] = {}


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


async def run_pipeline(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    media_type: str,
    file_id: str | None = None,
) -> None:
    msg = update.message
    chat = update.effective_chat
    initial_state = build_pipeline_state(update, context, media_type, file_id)

    try:
        pipeline = agent.get_pipeline()
        final_state = await pipeline.ainvoke(initial_state)
        response = final_state.get("response") or ""
        if response.strip():
            notification_msg = final_state.get("search_notification_msg")
            if notification_msg:
                await notification_msg.edit_text(strip_markdown(response))
            else:
                await msg.chat.send_action("typing")
                await msg.reply_text(strip_markdown(response))
    except DailyLimitError:
        logger.warning("Daily token quota exhausted for chat %s", chat.id)
        await msg.reply_text(
            "📵 Суточный лимит токенов Groq исчерпан. Бот ушёл спать до завтра. "
            "Статья на Луркоморье: «Бесплатный тариф — он такой»."
        )
    except RateLimitError:
        logger.warning("Rate limit reached for chat %s", chat.id)
        await msg.reply_text(
            "⏳ Groq не завезли лимитов. Бот временно на перекуре — слишком много запросов. "
            "Попробуйте через минуту, анончики."
        )
    except Exception as error:
        logger.error("Pipeline error for chat %s: %s", chat.id, error)
        await msg.reply_text(
            "Что-то сломалось. Скорее всего, Groq опять тупит. Попробуй позже."
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
    await notify_unlocks(context, chat_id, user_id, username)



async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    bot_id = context.bot.id
    text = update.message.text
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    await track_text_stats(update, context, user_id, chat_id, username, text)

    if is_reply_to_game_message(update):
        return

    reply = update.message.reply_to_message
    is_reply_to_bot = reply and reply.from_user and reply.from_user.id == bot_id

    # Offense auto-roast: if user insults the bot twice in a row.
    if OFFENSE_RE.search(text) and is_reply_to_bot:
        counts = offense_reply_counts.setdefault(chat_id, {})
        counts[user_id] = counts.get(user_id, 0) + 1
        if counts[user_id] >= 2:
            counts[user_id] = 0
            await update.message.chat.send_action("typing")
            try:
                roast_text = await generate_roast_text(chat_id, user_id, username)
                await update.message.reply_text(
                    f"{strip_markdown(roast_text)}\n\n#прожарка"
                )
                await achievements.increment_stat(user_id, chat_id, username, "roasted_count")
                await notify_unlocks(context, chat_id, user_id, username)
            except Exception as error:
                logger.error("Offense roast failed for %s in chat %s: %s", username, chat_id, error)
            return

    await run_pipeline(update, context, media_type="text")


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return
    if update.effective_user and update.effective_user.is_bot:
        return

    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    logger.info("Voice/video_note from @%s in chat %s", username, chat_id)

    if msg.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
        await notify_unlocks(context, chat_id, user_id, username)
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

    await notify_unlocks(context, chat_id, user_id, username)
    await run_pipeline(update, context, media_type=media_type, file_id=file_id)


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.photo:
        return
    if update.effective_user and update.effective_user.is_bot:
        return

    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if msg.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
        await notify_unlocks(context, chat_id, user_id, username)
        return

    await achievements.increment_stat(user_id, chat_id, username, "photo_messages")
    await notify_unlocks(context, chat_id, user_id, username)

    photo = msg.photo[-1]
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
        await notify_unlocks(context, chat_id, user_id, username)
        return
    await achievements.increment_stat(user_id, chat_id, username, "sticker_messages")
    await notify_unlocks(context, chat_id, user_id, username)


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
        await notify_unlocks(context, chat_id, user_id, username)
        return
    await achievements.increment_stat(user_id, chat_id, username, "animation_messages")
    await notify_unlocks(context, chat_id, user_id, username)


async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pass


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
        await notify_unlocks(context, chat_id, user_id, username)
        return
    await achievements.increment_stat(user_id, chat_id, username, "video_messages")
    await notify_unlocks(context, chat_id, user_id, username)
    await run_pipeline(update, context, media_type="video", file_id=msg.video.file_id)
