"""
Entry point for the Telegram bot.
Run with: python -m src.bot
"""

import datetime
import io
import logging
import random
import re

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

from src import achievements, config, psstore, ranks, wishlist
from src.agent import DailyLimitError, RateLimitError, init_agent, run_agent, run_lightweight
from src.memory import get_chat_history, get_recent_messages

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

GAME_KEYWORDS = re.compile(
    r"(игр[аыуеёюеи]?|поигра|сыграем|зайдёшь|зайдешь|онлайн|кооп|кросплей|crossplay|"
    r"ps5|playstation|стим|steam|мультиплеер|multiplayer|лобби|lobby|рейтинг|rank|"
    r"апдейт|update|патч|patch|длс|dlc|сервер|server|лаги|lag|тайтл|релиз|release|"
    r"геймплей|gameplay|открытый мир|open world|шутер|shooter|рпг|rpg|мморпг|mmorpg|"
    r"fps|фпс|frame rate|ray tracing|рейтрейсинг|gpu|cpu|vram|nvme|latency|пинг|"
    r"разрешение|resolution|4k|1080p|1440p|dlss|fsr|upscaling|апскейлинг|герц|hz|"
    r"dualsense|haptic|адаптивные триггеры|adaptive triggers)",
    re.IGNORECASE | re.UNICODE,
)

TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")

GAMES_PROMPTS = [
    (
        "Используй find_new_ps5_online_games(21) чтобы найти свежие PS5 релизы для онлайна за последние 3 недели. "
        "Для 2-3 самых интересных — проверь мультиплеер через get_game_details и живость через get_steam_player_count. "
        "Обязательно укажи: есть ли кросплей с PC, сколько игроков поддерживается онлайн."
    ),
    (
        "Используй get_ps_store_sales(15) чтобы найти текущие скидки в PS Store. "
        "Из этого списка отбери 2-3 сетевые игры для PS5 и проверь их через search_games + get_game_details. "
        "Для каждой расскажи: кросплей с PC есть или нет, сколько игроков онлайн, стоит ли брать по скидке."
    ),
    (
        "Двойной удар: используй find_new_ps5_online_games(30) для новинок и get_ps_store_sales(15) для скидок. "
        "Найди пересечение — есть ли свежие игры со скидкой? Если нет, дай по одному примеру из каждой категории. "
        "Для каждой игры — кросплей с PC и максимальное число игроков онлайн."
    ),
    (
        "Используй find_new_ps5_online_games(60) чтобы найти PS5 онлайн-игры за последние 2 месяца. "
        "Для топ-3 по рейтингу проверь get_steam_player_count — реально ли живые? "
        "Добавь: кросплей с PC через get_game_details. Итог: что сейчас реально стоит запускать."
    ),
]

PLAY_QUESTIONS_WITH_GAME = [
    "Кто играет сегодня в {game}?",
    "Сегодняшняя сессия по {game} — кто идёт?",
    "Собираем отряд в {game} — ты как?",
    "{game} сегодня — кто в деле?",
    "Кто готов страдать в {game} этим вечером?",
    "Залетаем в {game}? Отмечайтесь.",
]
PLAY_QUESTIONS_NO_GAME = [
    "Кто играет сегодня вечером?",
    "Собираем лобби — кто в деле?",
    "Игровая сессия сегодня — ты как?",
    "Кто готов страдать этим вечером?",
    "Вечерний гейминг — отмечайтесь.",
    "Залетаем сегодня? Кто есть?",
]

PLAY_OPTION_SETS = [
    ["Я в деле 🎮", "Может быть 🤔", "Не смогу 😢"],
    ["Врываюсь 🔥", "Подумаю 🤷", "Пасс 🙅"],
    ["Готов 👾", "Возможно 🎲", "Занят 😴"],
    ["Буду! 🎯", "Может чуть позже ⏰", "Без меня 🫡"],
    ["Уже качаю 📥", "Посмотрим 👀", "Нет сил 💀"],
    ["Первым в лобби 🏆", "Скорее всего ✅", "Нет 🚫"],
    ["Да! 🙌", "Буду поздно 🌙", "Только посмотреть 👁", "Пасс 💨"],
    ["Врываюсь 🚀", "Приду чуть позже ⏰", "Может быть 🤔", "Не сегодня 😵"],
    ["Уже в лобби 🟢", "Ещё думаю 🟡", "Не могу 🔴"],
    ["ГГ 🏅", "Может быть 🃏", "АФК сегодня 💤"],
    ["Да, точно 💪", "Постараюсь 🤞", "Вряд ли 😬", "Точно нет ❌"],
    ["Готов к бою ⚔️", "Залечу позже 🌙", "Пасс 🛌"],
]

MOSCOW_TZ = datetime.timezone(datetime.timedelta(hours=3))

AUTONOMOUS_COOLDOWN_SECONDS = 60
ROAST_COOLDOWN_SECONDS = 120
ROAST_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3-turbo"
VOICE_RESPONSE_CHANCE = 0.5
MAX_ROASTS_PER_USER_PER_DAY = 2

# Track per-user roast count per day per chat: {chat_id: {date_str: {username: count}}}
__daily_roast_counts: dict[int, dict[str, dict[str, int]]] = {}

ROAST_WORLD_THEMES = [
    "который купил NFT на последние деньги",
    "который ждёт реstock геймпада уже третий год",
    "который объясняет маме что такое battle royale",
    "который скачивает 150GB обновление по мобильному интернету",
    "который проиграл 5 раз подряд и винит пинг",
    "который читает гайд как пройти туториал",
    "который покупает 99 DLC к игре за полную цену",
    "который пытается найти трёх друзей для кооп-игры в 2 часа ночи",
    "который пропустил старт продаж новой консоли",
    "который требует кросплей с PS2",
]

ROULETTE_ANNOUNCEMENTS = [
    "🔫 Русская рулетка! Барабан крутится... @{username} — *БАХ!* 💀 Сегодня не твой день. Бывает.",
    "🔫 Три... два... один... *ВЫСТРЕЛ!* @{username} поймал пулю. Ничего личного — просто статистика.",
    "🎰 Рулетка выбрала жертву: @{username}. 🔫💥 Удача — дама непостоянная, увы.",
    "🔫 Барабан долго крутился и остановился на @{username}. Завтра лучше будет. Наверное.",
    "🎲 Судьба сегодня выбрала @{username}. 🔫 Не нам судить волю рулетки.",
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


def __is_night_message(update: Update) -> bool:
    """True if the message was sent between 00:00 and 05:00 Moscow time."""
    if not update.message or not update.message.date:
        return False
    moscow_time = update.message.date.astimezone(MOSCOW_TZ)
    return 0 <= moscow_time.hour < 5


def __parse_play_args(args: list[str]) -> tuple[datetime.datetime | None, str | None]:
    """Returns (reminder_time, game_name) parsed from /play command args."""
    if not args:
        return None, None

    reminder_time = None
    game_parts = list(args)

    if TIME_PATTERN.match(args[0]):
        hour, minute = map(int, args[0].split(":"))
        now = datetime.datetime.now(MOSCOW_TZ)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += datetime.timedelta(days=1)
        reminder_time = target
        game_parts = args[1:]

    game_name = " ".join(game_parts) if game_parts else None
    return reminder_time, game_name


async def __send_agent_reply(update: Update, username: str, message_text: str) -> None:
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


async def __send_lightweight_reply(update: Update, username: str, message_text: str) -> None:
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
        "/games — свежие игры для PS5: новинки, скидки, кросплей\n"
        "/coop — PS5 кооп-игра для 3-8 участников\n"
        "/play [ЧЧ:ММ] [игра] — опрос кто играет сегодня + напоминание\n"
        "/achievements [all] — достижения (свои или всех)\n"
        "/rank — твой ранг и сколько очков заработал\n"
        "/top — рейтинг всего чата\n"
        "/prozharka — случайный участник получает по заслугам\n\n"
        "Или просто упомяни меня через @ и задай любой вопрос."
    )


async def cmd_games(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    username = __get_username(update)
    await __send_agent_reply(update, username, random.choice(GAMES_PROMPTS))


async def cmd_coop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    username = __get_username(update)
    await achievements.increment_stat(
        update.effective_user.id, update.effective_chat.id, username, "coop_queries"
    )
    await __send_agent_reply(
        update,
        username,
        (
            "Найди одну подходящую игру для кооп-сессии чата на от 3 до 8 игроков. "
            "Используй find_coop_games(3) для поиска кандидатов на PS5. "
            "Выбери самый интересный вариант — PS5-эксклюзив или игра с кросплеем с PC. "
            "Расскажи про неё: название, жанр, максимальное число онлайн-игроков, есть ли кросплей с PC."
        ),
    )


async def cmd_play(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    reminder_time, game_name = __parse_play_args(context.args or [])

    if game_name:
        poll_question = random.choice(PLAY_QUESTIONS_WITH_GAME).format(game=game_name)
    else:
        poll_question = random.choice(PLAY_QUESTIONS_NO_GAME)

    await update.message.reply_poll(
        question=poll_question,
        options=random.choice(PLAY_OPTION_SETS),
        is_anonymous=False,
    )

    username = __get_username(update)
    await achievements.increment_stat(
        update.effective_user.id, update.effective_chat.id, username, "play_polls_created"
    )

    if reminder_time:
        time_str = reminder_time.strftime("%H:%M")
        now = datetime.datetime.now(MOSCOW_TZ)
        delay = (reminder_time - now).total_seconds()
        context.job_queue.run_once(
            __session_reminder,
            when=delay,
            chat_id=update.effective_chat.id,
            data={"game": game_name},
        )
        game_label = f" {game_name}" if game_name else ""
        await update.message.reply_text(
            f"⏰ Напомню в {time_str} МСК{game_label}. Можете пока поспорить кто виноват в прошлом проигрыше."
        )


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

async def __session_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    game = context.job.data.get("game")
    text = (
        f"⏰ Время! Сессия по {game} начинается — все в лобби!"
        if game
        else "⏰ Время! Игровая сессия начинается — где все?"
    )
    await context.bot.send_message(chat_id=context.job.chat_id, text=text)


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
            f"как лучший друг, который реально верит в него. Без сарказма, с душой. До 3 предложений."
        )
    else:
        style_instruction = (
            f"Напиши жёсткий саркастический роаст на @{target_username} в стиле стендап-комика. "
            f"Максимум 3 предложения. Злой юмор, чёрный сарказм, смешно и больно. "
            f"Обязательно упомяни @{target_username} в тексте."
        )

    if history_text and random.random() < 0.5:
        context_line = f"Последние сообщения @{target_username} в чате:\n{history_text}"
    else:
        theme = random.choice(ROAST_WORLD_THEMES)
        context_line = f"Придумай роаст на @{target_username} {theme}."

    response = await llm.ainvoke([
        SystemMessage(content=(
            "Ты стендап-комик в группе друзей-геймеров. "
            "Пишешь короткие роасты — строго до 3 предложений. "
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
                f"Как лучший друг — искренне, без сарказма. До 3 предложений."
            )
        else:
            style_instruction = (
                f"Напиши жёсткий утренний роаст-прогноз для @{target_username} в стиле стендап-комика. "
                f"Максимум 3 предложения. Злой юмор, сарказм, смешно. "
                f"Упомяни @{target_username} в тексте."
            )

        if history_text and random.random() < 0.5:
            context_line = f"Последние сообщения @{target_username}:\n{history_text}"
        else:
            theme = random.choice(ROAST_WORLD_THEMES)
            context_line = f"Придумай роаст на @{target_username} {theme}."

        try:
            response = await llm.ainvoke([
                SystemMessage(content=(
                    "Ты стендап-комик в группе друзей-геймеров. "
                    "Пишешь короткие утренние роасты — строго до 3 предложений. "
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
        template = random.choice(ROULETTE_ANNOUNCEMENTS)
        message = template.format(username=victim_username)
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
    if not __should_respond(update, bot_id):
        return

    is_direct = __is_bot_mentioned(update) or __is_reply_to_bot(update, bot_id)

    if not is_direct:
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
        await __send_agent_reply(update, username, update.message.text)
    else:
        await __send_lightweight_reply(update, username, update.message.text)


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
    is_group = update.effective_chat.type in ("group", "supergroup")
    if is_group and random.random() > VOICE_RESPONSE_CHANCE:
        return

    msg = update.message
    if not msg:
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
        bot_response = await run_lightweight(chat_id, username, transcript)
        formatted = __to_telegram_md(bot_response)
        try:
            await msg.reply_text(formatted, parse_mode="Markdown")
        except BadRequest:
            await msg.reply_text(bot_response)
        await achievements.increment_interaction(update.effective_user.id, numeric_chat_id, username)
    except (DailyLimitError, RateLimitError):
        logger.warning(f"Voice reply skipped (quota) for chat {chat_id}")
    except Exception as error:
        logger.error(f"Voice reply error in chat {chat_id}: {error}")


# --- Startup / entry point ---

async def __on_startup(application: Application) -> None:
    await init_agent()
    await wishlist.init_tables()
    await achievements.init_tables()
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
    app.add_handler(CommandHandler("games", cmd_games))
    app.add_handler(CommandHandler("coop", cmd_coop))
    app.add_handler(CommandHandler("play", cmd_play))
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

    logger.info("Starting polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
