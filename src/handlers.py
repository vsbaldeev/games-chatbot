"""
Telegram update handlers.

Achievement tracking lives here.  All AI response logic (routing, transcription,
vision, context assembly, LLM invocation, memory writing) is delegated to the
LangGraph pipeline via __run_pipeline().

Special cases that bypass the pipeline:
  - Offense auto-roast (OFFENSE_RE) remains here to avoid adding pipeline state.
  - Photo reality pre-check (is it a real photograph?) runs before the pipeline.
"""

import base64
import io
import re
from src import log

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from telegram import Update
from telegram.ext import ContextTypes

from src import achievements, config
from src.agent import agent, DailyLimitError, RateLimitError
from src.helpers import (
    fallback_username,
    get_username,
    is_night_message,
    is_reply_to_game_message,
    notify_unlocks,
    strip_markdown,
    OFFENSE_RE,
)
from src.pipeline.state import BotState, IncomingMessage
from src.prozharka import generate_prozharka_text

logger = log.get_logger(__name__)

WHISPER_MODEL = "whisper-large-v3"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

__EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027FF\U0001FA00-\U0001FAFF]",
    re.UNICODE,
)
__URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# {chat_id: {user_id: count}} — tracks consecutive offensive replies toward the bot
__offense_reply_counts: dict[int, dict[int, int]] = {}


async def __run_pipeline(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    media_type: str,
    file_id: str | None = None,
) -> None:
    """Build BotState and run the LangGraph pipeline. Sends reply if one is produced."""
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
    initial_state: BotState = {
        "incoming": incoming,
        "should_respond": False,
        "context": None,
        "response": None,
        "context_types": context,
    }

    try:
        pipeline = agent.get_pipeline()
        final_state = await pipeline.ainvoke(initial_state)
        response = final_state.get("response") or ""
        if response.strip():
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


async def track_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_chat:
        return
    user = update.effective_user
    username = user.username or user.first_name or fallback_username(user.id)
    await achievements.register_member(update.effective_chat.id, user.id, username)

    if update.message and not user.is_bot:
        await achievements.set_message_author(
            update.effective_chat.id, update.message.message_id, user.id, username
        )


async def handle_new_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.new_chat_members:
        return
    chat_id = update.effective_chat.id
    for user in update.message.new_chat_members:
        if user.is_bot:
            continue
        username = user.username or user.first_name or fallback_username(user.id)
        await achievements.register_member(chat_id, user.id, username)


async def handle_bot_added_to_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Seed chat_members with current admins when the bot is added to a group."""
    if not update.my_chat_member:
        return
    new_status = update.my_chat_member.new_chat_member.status
    if new_status not in ("member", "administrator"):
        return
    chat_id = update.effective_chat.id
    try:
        admins = await context.bot.get_chat_administrators(chat_id)
        for admin in admins:
            if admin.user.is_bot:
                continue
            username = admin.user.username or admin.user.first_name or fallback_username(admin.user.id)
            await achievements.register_member(chat_id, admin.user.id, username)
        logger.info("Seeded %d admins for chat %s on bot join", len(admins), chat_id)
    except Exception as error:
        logger.warning("Failed to seed admins for chat %s: %s", chat_id, error)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    bot_id = context.bot.id
    text = update.message.text
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Achievement tracking.
    if is_night_message(update):
        await achievements.increment_stat(user_id, chat_id, username, "night_messages")
    if __EMOJI_RE.search(text):
        await achievements.increment_stat(user_id, chat_id, username, "emoji_messages")
    if __URL_RE.search(text):
        await achievements.increment_stat(user_id, chat_id, username, "link_messages")
    if update.message.forward_origin is not None:
        await achievements.increment_stat(user_id, chat_id, username, "forwarded_messages")
    await achievements.update_max_stat(user_id, chat_id, username, "long_message_max", len(text))
    await notify_unlocks(context, chat_id, user_id, username)

    if is_reply_to_game_message(update):
        return

    reply = update.message.reply_to_message
    is_reply_to_bot = reply and reply.from_user and reply.from_user.id == bot_id

    # Offense auto-roast: if user insults the bot twice in a row.
    if OFFENSE_RE.search(text) and is_reply_to_bot:
        counts = __offense_reply_counts.setdefault(chat_id, {})
        counts[user_id] = counts.get(user_id, 0) + 1
        if counts[user_id] >= 2:
            counts[user_id] = 0
            await update.message.chat.send_action("typing")
            try:
                prozharka_text = await generate_prozharka_text(chat_id, username)
                await update.message.reply_text(
                    f"🔥 Прожарка @{username}:\n\n{strip_markdown(prozharka_text)}"
                )
                await achievements.increment_stat(user_id, chat_id, username, "roasted_count")
                await notify_unlocks(context, chat_id, user_id, username)
            except Exception as error:
                logger.error("Offense prozharka failed for %s in chat %s: %s", username, chat_id, error)
            return

    await __run_pipeline(update, context, media_type="text")


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
    await __run_pipeline(update, context, media_type=media_type, file_id=file_id)


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

    # Pre-filter: only respond to real photographs (not memes/screenshots/game art).
    if not await __is_real_photo(photo.file_id, context.bot):
        logger.info("Photo skipped (not real photo) in chat %s", chat_id)
        return

    await __run_pipeline(update, context, media_type="photo", file_id=photo.file_id)


async def __is_real_photo(file_id: str, bot) -> bool:
    """Return True if the image appears to be a real photograph taken with a camera."""
    try:
        tg_file = await bot.get_file(file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(buffer)
        raw_bytes = buffer.getvalue()

        if raw_bytes[:4] == b'\x89PNG':
            mime = "image/png"
        elif raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
            mime = "image/webp"
        else:
            mime = "image/jpeg"

        b64_image = base64.b64encode(raw_bytes).decode()
        image_url = f"data:{mime};base64,{b64_image}"

        llm = ChatGroq(
            model=VISION_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=5,
        )
        check = await llm.ainvoke([
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": (
                    "Is this a real photograph taken by a person with a camera "
                    "(photo of real life, people, places, objects, setups)? "
                    "Answer only YES or NO."
                )},
            ]),
        ])
        return check.content.strip().upper().startswith("YES")
    except Exception as error:
        logger.warning("Photo reality check failed for file %s: %s", file_id, error)
        return False


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


async def handle_reaction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    reaction = update.message_reaction
    if not reaction:
        return

    old_emojis = {r.emoji for r in reaction.old_reaction if hasattr(r, "emoji")}
    added_emojis = {
        r.emoji for r in reaction.new_reaction
        if hasattr(r, "emoji") and r.emoji not in old_emojis
    }
    if not added_emojis:
        return

    chat_id = reaction.chat.id
    author = await achievements.get_message_author(chat_id, reaction.message_id)
    if not author:
        logger.warning(
            "No author cached for message %s in chat %s — skipping reaction",
            reaction.message_id, chat_id,
        )
        return
    author_id, author_username = author
    if reaction.user and reaction.user.id == author_id:
        return
    logger.info(
        "Reaction %s on message %s in chat %s credited to %s",
        added_emojis, reaction.message_id, chat_id, author_username,
    )

    __LAUGH_EMOJIS = {"😁", "🤣"}
    __HEART_EMOJIS = {"❤", "🥰", "😍", "💘", "❤️‍\U0001f525"}
    __FIRE_EMOJIS  = {"🔥"}
    __THUMB_EMOJIS = {"👍"}

    stat_map = [
        (__LAUGH_EMOJIS, "laugh_reactions"),
        (__HEART_EMOJIS, "heart_reactions"),
        (__FIRE_EMOJIS,  "fire_reactions"),
        (__THUMB_EMOJIS, "thumbsup_reactions"),
    ]
    credited_any = False
    for emoji_set, stat_name in stat_map:
        if added_emojis & emoji_set:
            await achievements.increment_stat(author_id, chat_id, author_username, stat_name)
            credited_any = True

    if credited_any:
        await notify_unlocks(context, chat_id, author_id, author_username)
