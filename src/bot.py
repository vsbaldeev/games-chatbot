"""
Entry point for the Telegram bot.
Run with: python -m src.bot
"""

import base64
import datetime
import io
import json
import logging
import random
import re
from pathlib import Path

from groq import AsyncGroq
from telegram.error import BadRequest

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import Update
from telegram.helpers import escape_markdown
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from src import achievements, config, game_tracker, psstore, ranks, wishlist
from src.agent import LIGHTWEIGHT_MODEL, SYSTEM_PROMPT, DailyLimitError, RateLimitError, init_agent, run_agent, run_lightweight
from src.memory import get_chat_history, get_recent_messages

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

GAME_KEYWORDS = re.compile(
    r"(игр[аыуеёюеи]?|поигра|сыграем|зайдёшь|зайдешь|онлайн|кооп|кросплей|crossplay|"
    r"ps5|playstation|стим|пс|steam|мультиплеер|multiplayer|лобби|lobby|рейтинг|rank|"
    r"апдейт|update|патч|patch|длс|dlc|длс|сервер|server|лаги|lag|тайтл|релиз|release|"
    r"геймплей|gameplay|открытый мир|open world|шутер|shooter|рпг|rpg|мморпг|mmorpg|"
    r"fps|фпс|frame rate|ray tracing|рейтрейсинг|gpu|cpu|vram|nvme|latency|пинг|"
    r"разрешение|resolution|4k|1080p|1440p|dlss|fsr|upscaling|апскейлинг|герц|hz|"
    r"joystick|джойстик|gamepad|геймпад|adaptive|triggers|триггер|"
    r"dualsense|haptic|адаптивные триггеры|adaptive triggers)",
    re.IGNORECASE | re.UNICODE,
)

TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")

OFFENSE_RE = re.compile(
    r"(тупой|тупая|тупит|идиот|дебил|мудак|г[ао]вн[оа]|хуйн[яе]|нахуй|пиздец|"
    r"отстой|бесполезн|сломан|не работает|глупый|глупая|дерьм[оа]|придур|долбо|"
    r"ёбан|еба[нл]|заткн|иди нах|иди в|stupid|useless|broken|dumb|trash|"
    r"garbage|sucks|piece of shit|fuck)",
    re.IGNORECASE | re.UNICODE,
)

BOT_REFERENCE_RE = re.compile(
    r"(бот[аеуыь]?|bot)",
    re.IGNORECASE | re.UNICODE,
)

_DAL_WORDS_PATH = Path(__file__).parent / "dal_words.json"
_DAL_WORDS: list[str] = json.loads(_DAL_WORDS_PATH.read_text(encoding="utf-8"))

MOSCOW_TZ = datetime.timezone(datetime.timedelta(hours=3))

AUTONOMOUS_COOLDOWN_SECONDS = 60
ROAST_COOLDOWN_SECONDS = 120
ROAST_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3-turbo"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VOICE_RESPONSE_CHANCE = 0.25
PHOTO_RESPONSE_CHANCE = 0.25
MAX_ROASTS_PER_USER_PER_DAY = 2

# Track per-user roast count per day per chat: {chat_id: {date_str: {username: count}}}
__daily_roast_counts: dict[int, dict[str, dict[str, int]]] = {}


ROULETTE_HIT = [
    "🔫 Барабан крутится... @{username} — *БАХ!* 💀 Сегодня не твой день. Бывает.",
    "🔫 Три... два... один... *ВЫСТРЕЛ!* @{username} поймал пулю. Ничего личного — просто статистика.",
    "🎰 Рулетка выбрала жертву: @{username}. 🔫💥 Удача — дама непостоянная, увы.",
    "🔫 Барабан остановился на @{username}. Завтра лучше будет. Наверное.",
    "🎲 Судьба выбрала @{username}. 🔫 Не нам судить волю рулетки.",
    "💀 @{username} — всё. Барабан не соврёт.",
    "🔫 Щёлк. *БАМ.* @{username} сегодня в минусе. Рандом не обсуждается.",
    "🎯 Прямо в @{username}. 🔫 Меткость — 100%, удача — 0%.",
]

ROULETTE_MISS = [
    "🔫 Барабан крутится... @{username}... *клик.* Осечка. Повезло, живи пока.",
    "😮‍💨 @{username} — *клик.* Пусто. Сегодня фартануло.",
    "🔫 Три... два... один... *клик.* @{username} выдохнул. Патрона не было.",
    "🎰 Барабан остановился на @{username}... *клик.* Нет патрона. В следующий раз.",
    "😅 @{username} смотрит в ствол... *клик.* Обошлось. Пока.",
    "🔫 @{username}. Пустой патронник. Рулетка решила пощадить — в этот раз.",
    "🍀 @{username} — *клик.* Мимо. Видимо, не судьба сегодня.",
]


def __get_today() -> str:
    return datetime.date.today().isoformat()


def __get_roast_count(chat_id: int, username: str) -> int:
    today = __get_today()
    return __daily_roast_counts.get(chat_id, {}).get(today, {}).get(username, 0)


def __record_roast(chat_id: int, username: str) -> None:
    today = __get_today()
    if chat_id not in __daily_roast_counts:
        __daily_roast_counts[chat_id] = {}
    for old_date in [date for date in __daily_roast_counts[chat_id] if date != today]:
        del __daily_roast_counts[chat_id][old_date]
    if today not in __daily_roast_counts[chat_id]:
        __daily_roast_counts[chat_id][today] = {}
    prev = __daily_roast_counts[chat_id][today].get(username, 0)
    __daily_roast_counts[chat_id][today][username] = prev + 1


def __to_telegram_md(text: str) -> str:
    """Sanitise LLM output for Telegram Markdown v1."""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)
    lines = [line for line in text.splitlines() if not TABLE_SEPARATOR_RE.match(line)]
    return "\n".join(lines)


# Matches lines that are intermediate tool-output artefacts: markdown headers,
# numbered list items, and indented sub-items (e.g. "   - Цена: ...")
_INTERMEDIATE_LINE_RE = re.compile(r"^(#{1,3} |\d+\. | {2,}- )")


def __extract_game_card(response: str) -> str:
    """Strip intermediate tool-result summaries, returning only the final game card."""
    lines = response.splitlines()

    # Find the last PS Store link line — it always closes the card.
    store_idx = next(
        (idx for idx in range(len(lines) - 1, -1, -1)
         if "🛒" in lines[idx] or "store.playstation.com" in lines[idx]),
        None,
    )
    if store_idx is None:
        return response

    # Walk backward from the store link until we hit an intermediate-output line.
    start_idx = store_idx
    for idx in range(store_idx - 1, -1, -1):
        if _INTERMEDIATE_LINE_RE.match(lines[idx]):
            break
        start_idx = idx

    # Drop any leading blank lines from the extracted block.
    result = lines[start_idx : store_idx + 1]
    while result and not result[0].strip():
        result = result[1:]

    return "\n".join(result).strip()


def fallback_username(user_id: int) -> str:
    return f"user_{user_id}"


def __get_username(update: Update) -> str:
    user = update.effective_user
    return user.username or user.first_name or fallback_username(user.id)


def __is_bot_mentioned(update: Update) -> bool:
    text = update.message.text or ""
    return config.BOT_USERNAME.lower() in text.lower()


def __is_reply_to_bot(update: Update, bot_id: int) -> bool:
    reply = update.message.reply_to_message
    if not reply or not reply.from_user:
        return False
    return reply.from_user.id == bot_id


def __should_respond(update: Update, bot_id: int) -> bool:
    text = update.message.text or ""
    return (
        __is_bot_mentioned(update)
        or __is_reply_to_bot(update, bot_id)
        or bool(GAME_KEYWORDS.search(text))
    )


def __is_offense_toward_bot(update: Update) -> bool:
    text = update.message.text or ""
    return bool(OFFENSE_RE.search(text)) and (
        bool(BOT_REFERENCE_RE.search(text)) or __is_bot_mentioned(update)
    )


def __is_night_message(update: Update) -> bool:
    """True if the message was sent between 00:00 and 05:00 Moscow time."""
    if not update.message or not update.message.date:
        return False
    moscow_time = update.message.date.astimezone(MOSCOW_TZ)
    return 0 <= moscow_time.hour < 5


async def __notify_unlocks(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    username: str,
) -> None:
    try:
        new_ach = await achievements.check_new_achievements(user_id, chat_id, username)
        for ach in new_ach:
            text = (
                f"🏆 @{username} получил достижение!\n\n"
                f"{ach.emoji} *{ach.title}*\n_{ach.description}_"
            )
            try:
                await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            except BadRequest:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🏆 {username}: {ach.emoji} {ach.title} — {ach.description}",
                )
        points, _, _ = await ranks.get_user_rank_info(user_id, chat_id)
        new_ranks = await achievements.check_new_ranks(user_id, chat_id, points, ranks.RANKS)
        if new_ranks:
            top = new_ranks[-1]
            text = f"⬆️ @{username} вышел на новый уровень — {top.emoji} *{top.title}*"
            try:
                await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            except BadRequest:
                await context.bot.send_message(
                    chat_id=chat_id, text=f"⬆️ {username}: {top.emoji} {top.title}"
                )
    except Exception as error:
        logger.warning(f"Achievement notification failed for user {user_id} in chat {chat_id}: {error}")


async def __send_agent_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, username: str, message_text: str) -> None:
    chat_id = str(update.effective_chat.id)
    user_id = update.effective_user.id
    numeric_chat_id = update.effective_chat.id

    await update.message.chat.send_action("typing")
    try:
        response = await run_agent(chat_id, username, message_text)
        formatted = __to_telegram_md(response)
        try:
            await update.message.reply_text(formatted, parse_mode="Markdown")
        except BadRequest:
            await update.message.reply_text(response)
        await achievements.increment_interaction(user_id, numeric_chat_id, username)
        await __notify_unlocks(context, numeric_chat_id, user_id, username)
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


async def __send_lightweight_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, username: str, message_text: str) -> None:
    """Reply via the lightweight model — used for keyword-triggered passive responses."""
    chat_id = str(update.effective_chat.id)
    user_id = update.effective_user.id
    numeric_chat_id = update.effective_chat.id

    await update.message.chat.send_action("typing")
    try:
        response = await run_lightweight(chat_id, username, message_text)
        formatted = __to_telegram_md(response)
        try:
            await update.message.reply_text(formatted, parse_mode="Markdown")
        except BadRequest:
            await update.message.reply_text(response)
        await achievements.increment_interaction(user_id, numeric_chat_id, username)
        await __notify_unlocks(context, numeric_chat_id, user_id, username)
    except (DailyLimitError, RateLimitError):
        # Keyword-triggered: user wasn't addressing the bot, so silently skip on quota issues.
        logger.warning(f"Lightweight reply skipped (quota) for chat {chat_id}")
    except Exception as error:
        logger.error(f"Lightweight reply error for chat {chat_id}: {error}")


# --- Command handlers ---

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет. Я здесь. Спрашивайте про игры — если, конечно, есть что спросить."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды:\n"
        "/multiplayer — одна кооп/онлайн игра PS5/PC с ценой в TRY\n"
        "/singleplayer — одна одиночная игра PS5/PC с ценой в TRY\n"
        "/achievements [all] — достижения (свои или всех)\n"
        "/rank — твой ранг и сколько очков заработал\n"
        "/top — рейтинг всего чата\n"
        "/prozharka — случайный участник получает по заслугам\n\n"
        "Или просто упомяни меня через @ и задай любой вопрос."
    )


async def __send_game_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    game_type: str,
    prompt: str,
) -> None:
    username = __get_username(update)
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    await update.message.chat.send_action("typing")
    try:
        response = await run_agent(str(chat_id), username, prompt)
        card = __extract_game_card(response)

        game_name = card.splitlines()[0].strip().lstrip("🎮").strip() if card else None

        await update.message.reply_text(card)

        if game_name:
            await game_tracker.mark_suggested(chat_id, game_name, game_type)

        await achievements.increment_interaction(user_id, chat_id, username)
        await __notify_unlocks(context, chat_id, user_id, username)
    except DailyLimitError:
        await update.message.reply_text(
            "📵 Суточный лимит токенов Groq исчерпан. Бот ушёл спать до завтра. "
            "Статья на Луркоморье: «Бесплатный тариф — он такой»."
        )
    except RateLimitError:
        await update.message.reply_text(
            "⏳ Groq не завезли лимитов. Слишком много запросов. Попробуй через минуту."
        )
    except Exception as error:
        logger.error(f"Game command ({game_type}) error for chat {chat_id}: {error}")
        await update.message.reply_text("Что-то сломалось. Попробуй позже.")


async def cmd_multiplayer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    excluded = await game_tracker.get_suggested(chat_id, "multiplayer")
    excluded_str = ", ".join(excluded) if excluded else "нет"
    offset = random.choice([0, 8, 16, 24])

    prompt = (
        "Вызови инструменты последовательно, без единого слова между ними: "
        f"1) find_coop_games(2, offset={offset}) — выбери одну PS5-игру с онлайн-коопом или мультиплеером, "
        f"которой НЕТ в этом списке: [{excluded_str}] "
        "2) get_game_details для выбранной игры — нужно для определения кросплея с PC по наличию PC в платформах "
        "3) get_ps_store_price_tr для выбранной игры. "
        "После всех инструментов выведи ТОЛЬКО этот блок — никаких заголовков, никаких промежуточных итогов, "
        "никаких нумерованных списков, только plain text:\n"
        "🎮 Название игры\n"
        "Жанр: ...\n"
        "Игроков онлайн: до N\n"
        "Кросплей с PC: Да / Нет / нет данных\n"
        "Цена в TRY: ... или нет данных\n"
        "Краткое описание на русском (1-2 предложения).\n"
        "🛒 https://store.playstation.com/tr-tr/..."
    )
    await __send_game_command(update, context, "multiplayer", prompt)


async def cmd_singleplayer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    excluded = await game_tracker.get_suggested(chat_id, "singleplayer")
    excluded_str = ", ".join(excluded) if excluded else "нет"
    offset = random.choice([0, 8, 16, 24])

    prompt = (
        "Вызови инструменты последовательно, без единого слова между ними: "
        f"1) find_singleplayer_ps_games(offset={offset}) — выбери одну игру для PS5, которой НЕТ в этом списке: [{excluded_str}] "
        "2) get_ps_store_price_tr для выбранной игры. "
        "После всех инструментов выведи ТОЛЬКО этот блок — никаких заголовков, никаких промежуточных итогов, "
        "никаких нумерованных списков, только plain text:\n"
        "🎮 Название игры\n"
        "Жанр: ...\n"
        "Цена в TRY: ... или нет данных\n"
        "Краткое описание на русском (1-2 предложения).\n"
        "🛒 https://store.playstation.com/tr-tr/..."
    )
    await __send_game_command(update, context, "singleplayer", prompt)



async def cmd_achievements(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    show_all = context.args and context.args[0].lower() == "all"
    chat_id = update.effective_chat.id
    username = __get_username(update)

    if show_all:
        summary = await achievements.get_chat_achievements_summary(chat_id)
        if not summary:
            await update.message.reply_text(
                "Ни у кого нет достижений. Либо все новенькие, либо слишком нормальные."
            )
            return
        lines = []
        for person, earned in summary.items():
            badges = " ".join(f"{achievement.emoji}{achievement.title}" for achievement in earned)
            lines.append(f"{person}: {badges}")
        await update.message.reply_text("Достижения компании:\n\n" + "\n".join(lines))
    else:
        earned = await achievements.get_user_achievements(update.effective_user.id, chat_id)
        if not earned:
            await update.message.reply_text(
                f"{username}, достижений нет. Либо ты новенький, либо слишком нормальный. Оба варианта подозрительны."
            )
            return
        safe_username = escape_markdown(username, version=1)
        lines = [
            f"{achievement.emoji} *{achievement.title}*\n_{achievement.description}_"
            for achievement in earned
        ]
        await update.message.reply_text(
            f"Достижения {safe_username}:\n\n" + "\n\n".join(lines),
            parse_mode="Markdown",
        )


async def cmd_rank(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    username = __get_username(update)
    chat_id = update.effective_chat.id

    points, rank, breakdown = await ranks.get_user_rank_info(user.id, chat_id)

    lines = [
        f"{rank.emoji} *{rank.title}*",
        f"⭐ {ranks.pluralize_points(points)}",
    ]
    if breakdown:
        lines.append("\nОткуда взялось:")
        lines.extend(breakdown)
    else:
        lines.append("\nПока ноль. Начни общаться с ботом — очки сами придут.")

    upcoming = ranks.next_rank(points)
    if upcoming:
        needed = upcoming.min_points - points
        lines.append(f"\nДо *{upcoming.title}* {upcoming.emoji}: ещё {ranks.pluralize_points(needed)}")
    else:
        lines.append("\n👑 Это вершина. Дальше некуда.")

    text = f"Ранг {username}:\n\n" + "\n".join(lines)
    formatted = __to_telegram_md(text)
    try:
        await update.message.reply_text(formatted, parse_mode="Markdown")
    except BadRequest:
        await update.message.reply_text(text)


async def cmd_top(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    board = await ranks.get_chat_leaderboard(chat_id)

    if not board:
        await update.message.reply_text("Нет данных. Пишите боту — зарабатывайте очки.")
        return

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    lines = []
    for position, (username, points, rank) in enumerate(board, 1):
        prefix = medals.get(position, f"{position}.")
        lines.append(
            f"{prefix} {rank.emoji} {username} — {ranks.pluralize_points(points)} ({rank.title})"
        )

    text = "🏆 Рейтинг чата:\n\n" + "\n".join(lines)
    formatted = __to_telegram_md(text)
    try:
        await update.message.reply_text(formatted, parse_mode="Markdown")
    except BadRequest:
        await update.message.reply_text(text)


async def cmd_prozharka(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id

    now = datetime.datetime.now().timestamp()
    last_prozharka = context.chat_data.get("last_prozharka_ts", 0.0)
    if now - last_prozharka < ROAST_COOLDOWN_SECONDS:
        remaining = int(ROAST_COOLDOWN_SECONDS - (now - last_prozharka))
        await update.message.reply_text(
            f"Прожарка только что была. Остынь — ещё {remaining} сек."
        )
        return

    members = await achievements.get_chat_members(chat_id)
    if not members:
        await update.message.reply_text(
            "В базе нет участников. Пусть сначала кто-нибудь напишет в чат."
        )
        return

    eligible = [
        (uid, uname) for uid, uname in members
        if __get_roast_count(chat_id, uname) < MAX_ROASTS_PER_USER_PER_DAY
    ]
    if not eligible:
        await update.message.reply_text(
            "Все участники уже получили своё сегодня. Возвращайтесь завтра."
        )
        return

    target_id, target_username = random.choice(eligible)

    await update.message.chat.send_action("typing")
    try:
        prozharka_text = await __generate_prozharka_text(chat_id, target_username)
        __record_roast(chat_id, target_username)
        context.chat_data["last_prozharka_ts"] = datetime.datetime.now().timestamp()
        formatted = __to_telegram_md(prozharka_text)
        try:
            await update.message.reply_text(
                f"🔥 Прожарка @{target_username}:\n\n{formatted}",
                parse_mode="Markdown",
            )
        except BadRequest:
            await update.message.reply_text(
                f"🔥 Прожарка @{target_username}:\n\n{prozharka_text}"
            )
    except Exception as error:
        logger.error(f"Prozharka failed for {target_username} in chat {chat_id}: {error}")
        await update.message.reply_text("Прожарка не задалась. Groq на перекуре — попробуй позже.")


# --- Job callbacks ---

async def __weekly_dal_word(context: ContextTypes.DEFAULT_TYPE) -> None:
    if datetime.datetime.now(MOSCOW_TZ).weekday() != 6:  # 6 = Sunday
        return
    chat_ids = await achievements.get_all_chat_ids()
    if not chat_ids:
        return

    word = random.choice(_DAL_WORDS)
    try:
        llm = ChatGroq(
            model=LIGHTWEIGHT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.85,
            max_tokens=160,
        )
        response = await llm.ainvoke([
            SystemMessage(content=(
                "Ты саркастичный бот-геймер в групповом чате друзей. "
                "Пишешь только по-русски. Короткий живой стиль."
            )),
            HumanMessage(content=(
                f"Слово из словаря Даля: «{word}».\n\n"
                "Напиши два абзаца без заголовков:\n"
                "1. Краткое определение этого слова (1 предложение) в духе словаря Даля.\n"
                "2. Саркастичный пример использования в контексте видеоигр (1 предложение).\n"
                "Никаких заголовков, никаких нумераций, просто два предложения через пустую строку."
            )),
        ])
        body = response.content.strip()
    except Exception as error:
        logger.warning(f"Dal word LLM call failed: {error}")
        body = ""

    text = f"📖 Слово недели:\n\n*{word}*"
    if body:
        text += f"\n\n{body}"

    for chat_id in chat_ids:
        try:
            await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        except Exception as send_error:
            logger.warning(f"Dal word send failed for chat {chat_id}: {send_error}")


async def __daily_play_suggestion(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = await achievements.get_all_chat_ids()
    if not chat_ids:
        return
    try:
        llm = ChatGroq(
            model=LIGHTWEIGHT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.9,
            max_tokens=80,
        )
        response = await llm.ainvoke([
            SystemMessage(content=(
                "Ты саркастичный бот-геймер в групповом чате друзей. "
                "Пишешь только по-русски. Короткие, живые фразы."
            )),
            HumanMessage(content=(
                "Напиши одно короткое вечернее сообщение-призыв поиграть сегодня онлайн вместе. "
                "Обязательно упомяни, что речь об онлайн-игре или совместной сессии. "
                "Стиль: непринуждённый, слегка саркастичный, как будто зовёшь друзей. "
                "Без эмодзи в начале, без кавычек. Одно-два предложения максимум."
            )),
        ])
        suggestion = response.content.strip()
    except Exception as error:
        logger.warning(f"Daily play suggestion LLM call failed: {error}")
        suggestion = "Кто сегодня в онлайн? Пишите."
    for chat_id in chat_ids:
        try:
            await context.bot.send_message(chat_id=chat_id, text=suggestion)
        except Exception as send_error:
            logger.warning(f"Daily play suggestion failed for chat {chat_id}: {send_error}")


async def __check_ps_store_sales(context: ContextTypes.DEFAULT_TYPE) -> None:
    sales_by_chat = await psstore.find_wishlist_sales()
    for chat_id, matches in sales_by_chat.items():
        lines = []
        for user_id, username, wished, sale_title in matches:
            lines.append(f"• {username} хотел: «{wished}» → {sale_title}")
            await achievements.increment_stat(user_id, chat_id, username, "sale_notifications")
        await context.bot.send_message(
            chat_id=chat_id,
            text="🛒 Игры из вишлистов сейчас на скидке в PS Store:\n\n" + "\n".join(lines),
        )


async def __get_user_history_text(chat_id: int, username: str) -> str:
    """Return recent messages by username as a plain text block."""
    history = get_chat_history(str(chat_id))
    recent = await get_recent_messages(history, 40)
    user_prefix = f"{username}:"
    user_messages = [
        msg.content for msg in recent
        if hasattr(msg, "content")
        and isinstance(msg.content, str)
        and msg.content.startswith(user_prefix)
    ]
    return "\n".join(user_messages)


async def __generate_prozharka_text(chat_id: int, target_username: str) -> str:
    """Call the LLM to produce an on-demand prozharka for a chat member."""
    llm = ChatGroq(
        model=ROAST_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.95,
        max_tokens=180,
    )
    history_text = await __get_user_history_text(chat_id, target_username)

    is_supportive = random.random() < 0.1

    if is_supportive:
        style_instruction = (
            f"Напиши искреннее тёплое поддерживающее сообщение для @{target_username} — "
            f"как лучший друг, который реально верит в него. Без сарказма, с душой. До 2 предложений."
        )
    else:
        style_instruction = (
            f"Напиши жёсткий саркастический роаст на @{target_username} в стиле стендап-комика. "
            f"Максимум 2 предложения. Злой юмор, чёрный сарказм, смешно и больно. "
            f"Обязательно упомяни @{target_username} в тексте."
        )

    if history_text and random.random() < 0.5:
        context_line = f"Последние сообщения @{target_username} в чате:\n{history_text}"
    else:
        context_line = f"Придумай оригинальную тему для роаста на @{target_username} — что-нибудь из жизни геймера или просто абсурдное."

    response = await llm.ainvoke([
        SystemMessage(content=(
            "Ты стендап-комик в группе друзей-геймеров. "
            "Пишешь короткие роасты — строго до 2 предложений. "
            "Можно использовать мат и крепкие выражения. "
            "Только русский язык."
        )),
        HumanMessage(content=(
            f"{context_line}\n\n{style_instruction}"
        )),
    ])
    return response.content


async def __daily_roast(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = await achievements.get_all_chat_ids()
    if not chat_ids:
        return

    llm = ChatGroq(
        model=ROAST_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.9,
        max_tokens=180,
    )

    for chat_id in chat_ids:
        members = await achievements.get_chat_members(chat_id)
        if not members:
            continue

        eligible = [
            (uid, uname) for uid, uname in members
            if __get_roast_count(chat_id, uname) < MAX_ROASTS_PER_USER_PER_DAY
        ]
        if not eligible:
            continue

        target_user_id, target_username = random.choice(eligible)
        history_text = await __get_user_history_text(chat_id, target_username)

        is_supportive = random.random() < 0.1
        if is_supportive:
            style_instruction = (
                f"Напиши тёплое утреннее поддерживающее сообщение для @{target_username}. "
                f"Как лучший друг — искренне, без сарказма. До 2 предложений."
            )
        else:
            style_instruction = (
                f"Напиши жёсткий утренний роаст-прогноз для @{target_username} в стиле стендап-комика. "
                f"Максимум 2 предложения. Злой юмор, сарказм, смешно. "
                f"Упомяни @{target_username} в тексте."
            )

        if history_text and random.random() < 0.5:
            context_line = f"Последние сообщения @{target_username}:\n{history_text}"
        else:
            context_line = f"Придумай оригинальную тему для роаста на @{target_username} — что-нибудь из жизни геймера или просто абсурдное."

        try:
            response = await llm.ainvoke([
                SystemMessage(content=(
                    "Ты стендап-комик в группе друзей-геймеров. "
                    "Пишешь короткие утренние роасты — строго до 2 предложений. "
                    "Можно использовать мат и крепкие выражения. "
                    "Только русский язык."
                )),
                HumanMessage(content=(
                    f"{context_line}\n\n{style_instruction}"
                )),
            ])
            __record_roast(chat_id, target_username)
            logger.info(f"Daily prozharka tokens for chat {chat_id}: {response.response_metadata.get('token_usage')}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🌅 Доброе утро, @{target_username}:\n\n{response.content}",
            )
        except Exception as error:
            logger.warning(f"Daily prozharka failed for chat {chat_id}: {error}")


async def __russian_roulette(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = await achievements.get_all_chat_ids()
    for chat_id in chat_ids:
        members = await achievements.get_chat_members(chat_id)
        if len(members) < 2:
            continue
        victim_id, victim_username = random.choice(members)
        shot = random.random() < 0.5
        pool = ROULETTE_HIT if shot else ROULETTE_MISS
        message = random.choice(pool).format(username=victim_username)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode="Markdown",
            )
        except Exception as error:
            logger.warning(f"Russian roulette failed for chat {chat_id}: {error}")


# --- Background tracking handler (runs before all others) ---

async def __track_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_chat:
        return
    user = update.effective_user
    username = user.username or user.first_name or fallback_username(user.id)
    await achievements.register_member(update.effective_chat.id, user.id, username)


# --- Message handler ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    bot_id = context.bot.id
    is_direct = __is_bot_mentioned(update) or __is_reply_to_bot(update, bot_id)
    is_offense = __is_offense_toward_bot(update)

    if not is_direct and not is_offense and not __should_respond(update, bot_id):
        return

    if not is_direct and not is_offense:
        now = datetime.datetime.now().timestamp()
        last = context.chat_data.get("last_auto_ts", 0.0)
        if now - last < AUTONOMOUS_COOLDOWN_SECONDS:
            return
        context.chat_data["last_auto_ts"] = now

    username = __get_username(update)
    if __is_night_message(update):
        await achievements.increment_stat(
            update.effective_user.id, update.effective_chat.id, username, "night_messages"
        )
    if is_direct:
        await __send_agent_reply(update, context, username, update.message.text)
    else:
        await __send_lightweight_reply(update, context, username, update.message.text)


# --- Voice / video-note handler ---

async def __transcribe_telegram_file(file_id: str, filename: str, bot) -> str:
    """Download a Telegram file and transcribe it via Groq Whisper."""
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


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return

    is_group = update.effective_chat.type in ("group", "supergroup")
    caption = (msg.caption or "").lower()
    bot_mentioned = config.BOT_USERNAME.lower() in caption
    if is_group and not bot_mentioned and random.random() > VOICE_RESPONSE_CHANCE:
        return

    if msg.voice:
        file_id = msg.voice.file_id
        filename = "voice.ogg"
    elif msg.video_note:
        file_id = msg.video_note.file_id
        filename = "video_note.mp4"
    else:
        return

    username = __get_username(update)
    chat_id = str(update.effective_chat.id)
    numeric_chat_id = update.effective_chat.id

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
            model=LIGHTWEIGHT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.8,
            max_tokens=256,
        )
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=transcript),
        ])
        formatted = __to_telegram_md(response.content)
        try:
            await msg.reply_text(formatted, parse_mode="Markdown")
        except BadRequest:
            await msg.reply_text(response.content)
        await achievements.increment_interaction(update.effective_user.id, numeric_chat_id, username)
        await __notify_unlocks(context, numeric_chat_id, update.effective_user.id, username)
    except Exception as error:
        logger.error(f"Voice reply error in chat {chat_id}: {error}")


# --- Photo handler ---

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.photo:
        return
    if update.effective_user and update.effective_user.is_bot:
        return
    if msg.forward_origin is not None:
        return

    is_group = update.effective_chat.type in ("group", "supergroup")
    caption = (msg.caption or "").lower()
    bot_mentioned = config.BOT_USERNAME.lower() in caption
    if is_group and not bot_mentioned and random.random() > PHOTO_RESPONSE_CHANCE:
        return

    # Telegram provides multiple sizes; the last entry is the largest.
    photo = msg.photo[-1]
    username = __get_username(update)
    chat_id = str(update.effective_chat.id)
    numeric_chat_id = update.effective_chat.id

    await msg.chat.send_action("typing")
    try:
        tg_file = await context.bot.get_file(photo.file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(buffer)
        b64_image = base64.b64encode(buffer.getvalue()).decode()

        llm = ChatGroq(
            model=VISION_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=5,
        )
        check = await llm.ainvoke([
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
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
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                {"type": "text", "text": f"{username}: {user_text}"},
            ]),
        ])
        formatted = __to_telegram_md(response.content)
        try:
            await msg.reply_text(formatted, parse_mode="Markdown")
        except BadRequest:
            await msg.reply_text(response.content)
        await achievements.increment_interaction(update.effective_user.id, numeric_chat_id, username)
        await __notify_unlocks(context, numeric_chat_id, update.effective_user.id, username)
    except Exception as error:
        logger.error(f"Photo reply error in chat {chat_id}: {error}")


# --- Startup / entry point ---

async def __on_startup(application: Application) -> None:
    await init_agent()
    await wishlist.init_tables()
    await achievements.init_tables()
    await game_tracker.init_tables()
    await psstore.init_sale_tracking()

    # PS Store sale check daily at 10:00 Moscow time (07:00 UTC)
    application.job_queue.run_daily(
        __check_ps_store_sales,
        time=datetime.time(hour=7, minute=0, tzinfo=datetime.timezone.utc),
    )
    # Daily roast at 09:00 Moscow time (06:00 UTC)
    application.job_queue.run_daily(
        __daily_roast,
        time=datetime.time(hour=6, minute=0, tzinfo=datetime.timezone.utc),
    )
    # Russian roulette daily at 15:00 Moscow time (12:00 UTC)
    application.job_queue.run_daily(
        __russian_roulette,
        time=datetime.time(hour=12, minute=0, tzinfo=datetime.timezone.utc),
    )
    # Evening play suggestion daily at 21:00 Moscow time (18:00 UTC)
    application.job_queue.run_daily(
        __daily_play_suggestion,
        time=datetime.time(hour=18, minute=0, tzinfo=datetime.timezone.utc),
    )
    # Weekly Dal word on Sundays at 12:00 Moscow time (09:00 UTC)
    application.job_queue.run_daily(
        __weekly_dal_word,
        time=datetime.time(hour=9, minute=0, tzinfo=datetime.timezone.utc),
    )
    logger.info("Bot started, all tables and jobs initialized")


def main() -> None:
    app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_TOKEN)
        .post_init(__on_startup)
        .build()
    )

    app.add_handler(TypeHandler(Update, __track_member), group=-1)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("multiplayer", cmd_multiplayer))
    app.add_handler(CommandHandler("singleplayer", cmd_singleplayer))
    app.add_handler(CommandHandler("achievements", cmd_achievements))
    app.add_handler(CommandHandler("rank", cmd_rank))
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("prozharka", cmd_prozharka))

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
            handle_message,
        )
    )
    app.add_handler(
        MessageHandler(
            (filters.VOICE | filters.VIDEO_NOTE) & (filters.ChatType.GROUPS | filters.ChatType.PRIVATE),
            handle_voice_message,
        )
    )
    app.add_handler(
        MessageHandler(
            filters.PHOTO & (filters.ChatType.GROUPS | filters.ChatType.PRIVATE),
            handle_photo_message,
        )
    )

    logger.info("Starting polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
