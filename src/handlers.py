import base64
import io
import logging
import random
import re

from groq import AsyncGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from src import achievements, config
from src.agent import SYSTEM_PROMPT, DailyLimitError, RateLimitError, run_agent
from src.duel import DUEL_CHALLENGE_RE, handle_duel_mention
from src.prozharka import generate_prozharka_text
from src.helpers import (
    fallback_username,
    get_username,
    is_bot_mentioned,
    is_night_message,
    is_reply_to_bot,
    notify_unlocks,
    to_telegram_md,
    OFFENSE_RE,
)

logger = logging.getLogger(__name__)

WHISPER_MODEL = "whisper-large-v3"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VOICE_REPLY_MODEL = "llama-3.3-70b-versatile"
VOICE_RESPONSE_CHANCE = 0.25
PHOTO_RESPONSE_CHANCE = 0.25

__VOICE_SYSTEM_PROMPT = (
    "Ты дружелюбный участник чата геймеров. Тебе передали расшифровку голосового сообщения. "
    "Отреагируй тепло и непринуждённо — как живой человек в компании друзей. "
    "Коротко, по-русски, без формальностей."
)

__EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027FF\U0001FA00-\U0001FAFF]",
    re.UNICODE,
)
__URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# {chat_id: {user_id: count}} — tracks consecutive offensive replies toward the bot
__offense_reply_counts: dict[int, dict[int, int]] = {}

__LAUGH_EMOJIS = {"😂", "😁", "😆", "😀", "😃"}
__HEART_EMOJIS = {"❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍", "🤎", "💕", "❤‍🔥"}
__FIRE_EMOJIS  = {"🔥"}
__THUMB_EMOJIS = {"👍"}


async def __send_agent_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, username: str, message_text: str) -> None:
    chat_id = str(update.effective_chat.id)
    user_id = update.effective_user.id
    numeric_chat_id = update.effective_chat.id

    await update.message.chat.send_action("typing")
    try:
        response = await run_agent(chat_id, username, message_text)
        formatted = to_telegram_md(response)
        try:
            await update.message.reply_text(formatted, parse_mode="Markdown")
        except BadRequest:
            await update.message.reply_text(response)
        await notify_unlocks(context, numeric_chat_id, user_id, username)
    except DailyLimitError:
        logger.warning(f"Daily token quota exhausted for chat {chat_id}")
        await update.message.reply_text(
            "📵 Суточный лимит токенов Groq исчерпан. Бот ушёл спать до завтра. "
            "Статья на Луркоморье: «Бесплатный тариф — он такой»."
        )
    except RateLimitError:
        logger.warning(f"Rate limit reached for chat {chat_id}")
        await update.message.reply_text(
            "⏳ Groq не завезли лимитов. Бот временно на перекуре — слишком много запросов. "
            "Попробуйте через минуту, анончики."
        )
    except Exception as error:
        logger.error(f"Agent error for chat {chat_id}: {error}")
        await update.message.reply_text(
            "Что-то сломалось. Скорее всего, Groq опять тупит. Попробуй позже."
        )


async def __transcribe_telegram_file(file_id: str, filename: str, bot) -> str:
    tg_file = await bot.get_file(file_id)
    buffer = io.BytesIO()
    await tg_file.download_to_memory(buffer)
    buffer.seek(0)
    audio_bytes = buffer.read()

    client = AsyncGroq(api_key=config.GROQ_API_KEY)
    transcription = await client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=WHISPER_MODEL,
    )
    return transcription.text.strip()


async def track_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_chat:
        return
    user = update.effective_user
    username = user.username or user.first_name or fallback_username(user.id)
    await achievements.register_member(update.effective_chat.id, user.id, username)


async def handle_new_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.new_chat_members:
        return
    chat_id = update.effective_chat.id
    for user in update.message.new_chat_members:
        if user.is_bot:
            continue
        username = user.username or user.first_name or fallback_username(user.id)
        await achievements.register_member(chat_id, user.id, username)


__WELCOME_MESSAGE = """Привет. Я здесь.

Чтобы всё работало правильно:

• *Напишите что-нибудь в чат* — каждый участник регистрируется, когда впервые пишет сообщение, реагирует или отправляет медиа. Только зарегистрированные попадают в рулетку и дуэли.
• *Отключите Privacy Mode* у бота (через BotFather → Bot Settings → Group Privacy → Turn off) — иначе я вижу только команды, но не обычные сообщения.

Что умею:
/duel — эмодзи-дуэль между двумя участниками
/ruletka — русская рулетка
/prozharka — прожарка случайного участника
/multiplayer — одна кооп/онлайн игра PS5 с ценой
/singleplayer — одна одиночная игра PS5 с ценой
/achievements — твои достижения
/top — топ чата

Ещё отвечаю на вопросы про игры, если упомянуть меня через @."""


async def handle_bot_added_to_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Seed chat_members with current admins and send a welcome message when the bot is added to a group."""
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
        logger.info(f"Seeded {len(admins)} admins for chat {chat_id} on bot join")
    except Exception as error:
        logger.warning(f"Failed to seed admins for chat {chat_id}: {error}")
    try:
        await context.bot.send_message(chat_id=chat_id, text=__WELCOME_MESSAGE, parse_mode="Markdown")
    except Exception as error:
        logger.warning(f"Failed to send welcome message to chat {chat_id}: {error}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    bot_id = context.bot.id
    is_direct = is_bot_mentioned(update) or is_reply_to_bot(update, bot_id)
    text = update.message.text
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

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

    if DUEL_CHALLENGE_RE.search(text) and "@" in text:
        duel_started = await handle_duel_mention(update, context)
        if duel_started:
            return

    if not is_direct:
        return

    if OFFENSE_RE.search(text) and is_reply_to_bot(update, bot_id):
        counts = __offense_reply_counts.setdefault(chat_id, {})
        counts[user_id] = counts.get(user_id, 0) + 1
        if counts[user_id] >= 2:
            counts[user_id] = 0
            await update.message.chat.send_action("typing")
            try:
                prozharka_text = await generate_prozharka_text(chat_id, username)
                formatted = to_telegram_md(prozharka_text)
                try:
                    await update.message.reply_text(
                        f"🔥 Прожарка @{username}:\n\n{formatted}",
                        parse_mode="Markdown",
                    )
                except BadRequest:
                    await update.message.reply_text(
                        f"🔥 Прожарка @{username}:\n\n{prozharka_text}"
                    )
                await achievements.increment_stat(user_id, chat_id, username, "roasted_count")
                await notify_unlocks(context, chat_id, user_id, username)
            except Exception as error:
                logger.error(f"Offense prozharka failed for {username} in chat {chat_id}: {error}")
            return

    if DUEL_CHALLENGE_RE.search(text) and "@" in text:
        await handle_duel_mention(update, context)
        return

    await __send_agent_reply(update, context, username, text)


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return
    if update.effective_user and update.effective_user.is_bot:
        return
    if msg.forward_origin is not None:
        return

    if msg.voice:
        file_id = msg.voice.file_id
        filename = "voice.ogg"
    elif msg.video_note:
        file_id = msg.video_note.file_id
        filename = "video_note.mp4"
    else:
        return

    username = get_username(update)
    chat_id = str(update.effective_chat.id)
    numeric_chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if msg.voice:
        await achievements.increment_stat(user_id, numeric_chat_id, username, "voice_messages")
        await achievements.update_max_stat(user_id, numeric_chat_id, username, "voice_max_duration", msg.voice.duration)
    else:
        await achievements.increment_stat(user_id, numeric_chat_id, username, "video_note_messages")
    await notify_unlocks(context, numeric_chat_id, user_id, username)

    if random.random() > VOICE_RESPONSE_CHANCE:
        return

    await msg.chat.send_action("typing")
    try:
        transcript = await __transcribe_telegram_file(file_id, filename, context.bot)
    except Exception as error:
        logger.error(f"Transcription failed in chat {chat_id}: {error}")
        return

    if not transcript:
        return

    try:
        llm = ChatGroq(
            model=VOICE_REPLY_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.8,
            max_tokens=256,
        )
        response = await llm.ainvoke([
            SystemMessage(content=__VOICE_SYSTEM_PROMPT),
            HumanMessage(content=f"{username}: {transcript}"),
        ])
        formatted = to_telegram_md(response.content)
        try:
            await msg.reply_text(formatted, parse_mode="Markdown")
        except BadRequest:
            await msg.reply_text(response.content)
        await notify_unlocks(context, numeric_chat_id, update.effective_user.id, username)
    except Exception as error:
        logger.error(f"Voice reply error in chat {chat_id}: {error}")


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.photo:
        return
    if update.effective_user and update.effective_user.is_bot:
        return
    if msg.forward_origin is not None:
        return

    username = get_username(update)
    user_id = update.effective_user.id
    numeric_chat_id = update.effective_chat.id
    chat_id = str(numeric_chat_id)

    await achievements.increment_stat(user_id, numeric_chat_id, username, "photo_messages")
    await notify_unlocks(context, numeric_chat_id, user_id, username)

    caption = (msg.caption or "").lower()
    bot_mentioned = config.BOT_USERNAME.lower() in caption
    if not bot_mentioned and random.random() > PHOTO_RESPONSE_CHANCE:
        return

    photo = msg.photo[-1]

    await msg.chat.send_action("typing")
    try:
        tg_file = await context.bot.get_file(photo.file_id)
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
        if not check.content.strip().upper().startswith("YES"):
            logger.info(f"Photo skipped (not real photo) in chat {chat_id}")
            return

        user_text = (
            msg.caption
            or "Прокомментируй это изображение в своём стиле — саркастично, по-геймерски, коротко."
        )

        llm_reply = ChatGroq(
            model=VISION_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.8,
            max_tokens=300,
        )
        response = await llm_reply.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": f"{username}: {user_text}"},
            ]),
        ])
        formatted = to_telegram_md(response.content)
        try:
            await msg.reply_text(formatted, parse_mode="Markdown")
        except BadRequest:
            await msg.reply_text(response.content)
        await notify_unlocks(context, numeric_chat_id, update.effective_user.id, username)
    except Exception as error:
        logger.error(f"Photo reply error in chat {chat_id}: {error}")


async def handle_sticker_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.sticker:
        return
    if update.effective_user and update.effective_user.is_bot:
        return
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    await achievements.increment_stat(user_id, chat_id, username, "sticker_messages")
    await notify_unlocks(context, chat_id, user_id, username)


async def handle_video_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.video:
        return
    if update.effective_user and update.effective_user.is_bot:
        return
    if msg.forward_origin is not None:
        return
    username = get_username(update)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    await achievements.increment_stat(user_id, chat_id, username, "video_messages")
    await notify_unlocks(context, chat_id, user_id, username)


async def handle_reaction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    reaction = update.message_reaction
    if not reaction or not reaction.user:
        return
    # Only count newly added reactions, not removed ones
    old_emojis = {react.emoji for react in reaction.old_reaction if hasattr(react, "emoji")}
    added_emojis = {
        react.emoji for react in reaction.new_reaction
        if hasattr(react, "emoji") and react.emoji not in old_emojis
    }
    if not added_emojis:
        return

    reactor = reaction.user
    reactor_username = reactor.username or reactor.first_name or fallback_username(reactor.id)
    chat_id = reaction.chat.id

    # The Telegram API only exposes the reactor in MessageReactionUpdated, not the original
    # message author. We credit the reactor's stats as a proxy for engagement.
    stat_map = [
        (__LAUGH_EMOJIS, "laugh_reactions"),
        (__HEART_EMOJIS, "heart_reactions"),
        (__FIRE_EMOJIS,  "fire_reactions"),
        (__THUMB_EMOJIS, "thumbsup_reactions"),
    ]
    for emoji_set, stat_name in stat_map:
        if added_emojis & emoji_set:
            await achievements.increment_stat(reactor.id, chat_id, reactor_username, stat_name)
            await notify_unlocks(context, chat_id, reactor.id, reactor_username)
