"""
Entry point for the Telegram bot.
Run with: python -m src.bot
"""

import datetime
import logging
import random
import re

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

from src import achievements, config, features, game_filters, psstore, ranks, wishlist
from src.agent import DailyLimitError, RateLimitError, init_agent, run_agent
from src.memory import get_chat_history, get_recent_messages

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

GAME_KEYWORDS = re.compile(
    r"\b(игр[аыуеёюеи]?|поигра|сыграем|зайдёшь|зайдешь|онлайн|кооп|кросплей|crossplay|"
    r"ps5|playstation|стим|steam|мультиплеер|multiplayer|лобби|lobby|рейтинг|rank|"
    r"апдейт|update|патч|patch|длс|dlc|сервер|server|лаги|lag|тайтл|релиз|release|"
    r"геймплей|gameplay|открытый мир|open world|шутер|shooter|рпг|rpg|мморпг|mmorpg|"
    r"fps|фпс|frame rate|ray tracing|рейтрейсинг|gpu|cpu|vram|nvme|latency|пинг|"
    r"разрешение|resolution|4k|1080p|1440p|dlss|fsr|upscaling|апскейлинг|герц|hz|"
    r"dualsense|haptic|адаптивные триггеры|adaptive triggers)\b",
    re.IGNORECASE | re.UNICODE,
)

TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")

# Poll question templates. {game} is substituted when a game name is provided.
# Each entry is a distinct agent prompt for /games — rotated randomly each call.
# All variants ask about PS5 online games and must cover crossplay + player count.
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

# Each entry is one complete set of poll options (2–4 items).
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

# Cooldown between autonomous (keyword-triggered) responses in a chat.
# Direct mentions and commands are never throttled.
AUTONOMOUS_COOLDOWN_SECONDS = 60

# Per-user cooldown on /feature to prevent token abuse.
FEATURE_COOLDOWN_SECONDS = 30

# Per-chat cooldown on /prozharka to limit token burn on demand.
ROAST_COOLDOWN_SECONDS = 120
ROAST_MODEL = "llama-3.3-70b-versatile"
__feature_last_used: dict[int, float] = {}


def __to_telegram_md(text: str) -> str:
    """Sanitise LLM output for Telegram Markdown v1.

    The LLM sometimes produces standard Markdown (**bold**, tables) despite
    the system prompt. This converts the most common offenders so parse_mode
    does not silently corrupt the message.
    """
    # **bold** → *bold*  (Telegram v1 uses single asterisk)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)
    # Drop table separator rows (|---|---| etc.) — Telegram renders them as raw pipes
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


def __should_respond(update: Update) -> bool:
    text = update.message.text or ""
    return __is_bot_mentioned(update) or bool(GAME_KEYWORDS.search(text))


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
        # Use Moscow time so users get the reminder at the clock time they expect.
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

    user_filter_map = await game_filters.get_filters(user_id, numeric_chat_id)
    filter_hint = game_filters.build_filter_hint(user_filter_map)

    await update.message.chat.send_action("typing")
    try:
        response = await run_agent(chat_id, username, message_text, filter_hint=filter_hint)
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


# --- Command handlers ---

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет. Я здесь. Спрашивайте про игры — если, конечно, есть что спросить."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды:\n"
        "/games — свежие игры для PS5: новинки, скидки, кросплей\n"
        "/chto_takoe <запрос> — что это? Игра, термин, технология — бот разберётся сам\n"
        "/crossplay <игра> — есть ли кросплей\n"
        "/players <игра> — сколько людей сейчас в Steam\n"
        "/coop <число> — PS5 игры с онлайн кооп на N игроков\n"
        "/play [ЧЧ:ММ] [игра] — опрос кто играет сегодня + напоминание\n"
        "/wish add|list|remove|all — вишлист игр\n"
        "/achievements [all] — достижения (свои или всех)\n"
        "/rank — твой ранг и сколько очков заработал\n"
        "/top — рейтинг всего чата\n"
        "/prozharka — случайный участник получает по заслугам\n"
        "/feature <запрос> — предложить фичу\n"
        "/features — список ожидающих фич\n\n"
        "Фильтры рекомендаций:\n"
        "/ban <игра> — никогда не предлагать эту игру\n"
        "/known <игра> — уже знаю/играю, не предлагать\n"
        "/unban <игра> — убрать из фильтров\n"
        "/myfilters — посмотреть свои фильтры\n\n"
        "Или просто упомяни меня или напиши про игры."
    )


async def cmd_games(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    username = __get_username(update)
    await __send_agent_reply(update, username, random.choice(GAMES_PROMPTS))


async def cmd_crossplay(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    game = " ".join(context.args) if context.args else None
    if not game:
        await update.message.reply_text("Укажи игру: /crossplay Elden Ring")
        return
    username = __get_username(update)
    await achievements.increment_stat(
        update.effective_user.id, update.effective_chat.id, username, "crossplay_queries"
    )
    await __send_agent_reply(
        update,
        username,
        f"Есть ли кросплей в игре {game}? Особенно интересует кросплей между PS5 и PC.",
    )


async def cmd_players(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    game = " ".join(context.args) if context.args else None
    if not game:
        await update.message.reply_text("Укажи игру: /players Fortnite")
        return
    username = __get_username(update)
    await __send_agent_reply(
        update,
        username,
        f"Сколько сейчас людей играет в {game} в Steam? Жива ли игра?",
    )


async def cmd_chto_takoe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args) if context.args else None
    if not query:
        await update.message.reply_text(
            "Что объяснить? Пример: /chto_takoe DLSS или /chto_takoe Elden Ring"
        )
        return
    username = __get_username(update)
    await achievements.increment_stat(
        update.effective_user.id, update.effective_chat.id, username, "research_queries"
    )
    await __send_agent_reply(
        update,
        username,
        f"Что такое {query}? "
        f"Сам разберись: если это технический термин или концепция (DLSS, ray tracing, fps, HDR и т.п.) — "
        f"объясни своими словами, просто и с примерами, без лишних API-запросов. "
        f"Если это игра — используй search_games + get_game_details, расскажи про платформы, "
        f"мультиплеер, кросплей и живость по get_steam_player_count. "
        f"Отвечай коротко и по делу.",
    )


async def cmd_coop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Укажи количество игроков: /coop 4")
        return
    player_count = int(context.args[0])
    if not 2 <= player_count <= 32:
        await update.message.reply_text("Число игроков должно быть от 2 до 32.")
        return
    username = __get_username(update)
    await achievements.increment_stat(
        update.effective_user.id, update.effective_chat.id, username, "coop_queries"
    )
    await __send_agent_reply(
        update,
        username,
        f"Найди PS5 игры с онлайн кооп на {player_count} игроков. Используй find_coop_games({player_count}). Дай краткий обзор лучших вариантов.",
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


async def cmd_wish(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Использование:\n"
            "/wish add <игра> — добавить в вишлист\n"
            "/wish list — твой вишлист\n"
            "/wish remove <игра> — убрать из вишлиста\n"
            "/wish all — вишлисты всех участников чата"
        )
        return

    subcommand = context.args[0].lower()
    game_name = " ".join(context.args[1:]) if len(context.args) > 1 else None
    user = update.effective_user
    chat_id = update.effective_chat.id
    username = __get_username(update)

    if subcommand == "add":
        if not game_name:
            await update.message.reply_text("/wish add <название игры>")
            return
        await wishlist.add_game(user.id, chat_id, username, game_name)
        await update.message.reply_text(
            f"«{game_name}» добавлена в вишлист. Ждём скидку или пиратку."
        )

    elif subcommand == "list":
        games = await wishlist.get_user_wishlist(user.id)
        if not games:
            await update.message.reply_text(
                "Вишлист пуст. Либо всё уже есть, либо в игры не играешь."
            )
        else:
            items = "\n".join(f"• {game}" for game in games)
            await update.message.reply_text(f"Твой вишлист:\n{items}")

    elif subcommand == "remove":
        if not game_name:
            await update.message.reply_text("/wish remove <название игры>")
            return
        removed = await wishlist.remove_game(user.id, game_name)
        if removed:
            await update.message.reply_text(f"«{game_name}» удалена из вишлиста.")
        else:
            await update.message.reply_text(
                f"«{game_name}» не найдена. Проверь название — регистр не важен."
            )

    elif subcommand == "all":
        chat_wishlists = await wishlist.get_chat_wishlists(chat_id)
        if not chat_wishlists:
            await update.message.reply_text("Ни у кого нет вишлиста. Коллектив аскетов.")
        else:
            lines = [
                f"{person}: {', '.join(f'«{game}»' for game in games)}"
                for person, games in chat_wishlists.items()
            ]
            await update.message.reply_text("Вишлисты компании:\n\n" + "\n".join(lines))

    else:
        await update.message.reply_text(
            f"Неизвестная команда «{subcommand}». Используй: add, list, remove, all"
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


async def cmd_feature(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    description = " ".join(context.args) if context.args else None
    if not description:
        await update.message.reply_text(
            "Что предлагаешь? Пример: /feature уведомления о новых DLC"
        )
        return

    user_id = update.effective_user.id
    now = datetime.datetime.now().timestamp()
    last_used = __feature_last_used.get(user_id, 0.0)
    if now - last_used < FEATURE_COOLDOWN_SECONDS:
        await update.message.reply_text(
            "Полегче с запросами, анончик. Подожди немного перед следующим предложением."
        )
        return
    __feature_last_used[user_id] = now

    await update.message.chat.send_action("typing")
    already_exists = await features.check_if_implemented(description)
    if already_exists:
        await update.message.reply_text(
            f"Это уже есть, анончик. «{description}» — реализовано. "
            "Читай /help внимательнее, там всё написано."
        )
        return

    username = __get_username(update)
    request_id = await features.add_request(
        update.effective_chat.id, user_id, username, description
    )
    await update.message.reply_text(
        f"Записал запрос #{request_id}: «{description}». "
        "Когда выйдет обновление — бот сам объявит что завезли."
    )


async def cmd_features(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pending = await features.get_pending_for_chat(update.effective_chat.id)
    if not pending:
        await update.message.reply_text(
            "Ожидающих запросов нет. Либо всё уже сделано, либо анончики не просят."
        )
        return
    lines = [f"#{req.id} [{req.username}]: {req.description}" for req in pending]
    await update.message.reply_text(
        "Список ожидающих фич:\n\n" + "\n".join(lines)
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


async def cmd_ban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    game_name = " ".join(context.args) if context.args else None
    if not game_name:
        await update.message.reply_text("Какую игру баним? Пример: /ban Fortnite")
        return
    user = update.effective_user
    await game_filters.set_filter(user.id, update.effective_chat.id, game_name, game_filters.FILTER_BANNED)
    await update.message.reply_text(
        f"«{game_name}» — в чёрный список. Бот больше не будет это предлагать."
    )


async def cmd_unban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    game_name = " ".join(context.args) if context.args else None
    if not game_name:
        await update.message.reply_text("Что разбаниваем? Пример: /unban Fortnite")
        return
    user = update.effective_user
    removed = await game_filters.remove_filter(user.id, update.effective_chat.id, game_name)
    if removed:
        await update.message.reply_text(f"«{game_name}» — убрана из фильтров.")
    else:
        await update.message.reply_text(
            f"«{game_name}» не найдена в фильтрах. Проверь /myfilters."
        )


async def cmd_known(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    game_name = " ".join(context.args) if context.args else None
    if not game_name:
        await update.message.reply_text("Какую игру знаешь? Пример: /known Apex Legends")
        return
    user = update.effective_user
    await game_filters.set_filter(user.id, update.effective_chat.id, game_name, game_filters.FILTER_KNOWN)
    await update.message.reply_text(
        f"«{game_name}» — помечена как известная. Бот не будет предлагать в общих рекомендациях."
    )


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

    target_username = random.choice(members)[1]

    await update.message.chat.send_action("typing")
    try:
        prozharka_text = await __generate_prozharka_text(chat_id, target_username)
        context.chat_data["last_prozharka_ts"] = datetime.datetime.now().timestamp()
        formatted = __to_telegram_md(prozharka_text)
        try:
            await update.message.reply_text(
                f"🔥 Прожарка {target_username}:\n\n{formatted}",
                parse_mode="Markdown",
            )
        except BadRequest:
            await update.message.reply_text(
                f"🔥 Прожарка {target_username}:\n\n{prozharka_text}"
            )
    except Exception as error:
        logger.error(f"Prozharka failed for {target_username} in chat {chat_id}: {error}")
        await update.message.reply_text("Прожарка не задалась. Groq на перекуре — попробуй позже.")


async def cmd_myfilters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_filter_map = await game_filters.get_filters(user.id, update.effective_chat.id)
    banned = user_filter_map.get(game_filters.FILTER_BANNED, [])
    known = user_filter_map.get(game_filters.FILTER_KNOWN, [])
    if not banned and not known:
        await update.message.reply_text(
            "Фильтров нет. Используй /ban <игра> или /known <игра> чтобы добавить."
        )
        return
    lines = []
    if banned:
        lines.append("🚫 Забанено:\n" + "\n".join(f"  • {game}" for game in banned))
    if known:
        lines.append("✅ Уже знаю:\n" + "\n".join(f"  • {game}" for game in known))
    lines.append("\nУбрать фильтр: /unban <игра>")
    await update.message.reply_text("\n".join(lines))


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
    user_prefix = f"[{username}]:"
    user_messages = [
        msg.content for msg in recent
        if hasattr(msg, "content")
        and isinstance(msg.content, str)
        and msg.content.startswith(user_prefix)
    ]
    return "\n".join(user_messages)


async def __generate_prozharka_text(chat_id: int, username: str) -> str:
    """Call the LLM to produce an on-demand prozharka for a chat member."""
    llm = ChatGroq(
        model=ROAST_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.95,
        max_tokens=250,
    )
    history_text = await __get_user_history_text(chat_id, username)
    context_line = (
        f"Последние сообщения {username}:\n{history_text}"
        if history_text
        else f"{username} почти не писал — роастим по легенде, с фантазией."
    )
    response = await llm.ainvoke([
        SystemMessage(content=(
            "Ты бот в группе друзей-геймеров. Пишешь короткие (3-5 предложений) смешные роасты "
            "на участников чата на основе их активности. "
            "Стиль: дружеский стёб, как подкалывают друг друга в компании — остро, но без злобы и оскорблений. "
            "Только русский язык. Опирайся на конкретные темы из сообщений: игры, вопросы, привычки."
        )),
        HumanMessage(content=(
            f"Участник для роаста: {username}\n\n"
            f"{context_line}\n\n"
            f"Напиши смешной дружеский роаст на {username}."
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
        max_tokens=200,
    )

    for chat_id in chat_ids:
        members = await achievements.get_chat_members(chat_id)
        if not members:
            continue
        target_user_id, username = random.choice(members)

        history_text = await __get_user_history_text(chat_id, username)
        context_line = (
            f"Последние сообщения {username}:\n{history_text}"
            if history_text
            else f"{username} ещё не проявил себя в чате — пишем гороскоп вслепую."
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(content=(
                    "Ты бот в группе друзей-геймеров. Пишешь короткие (2-3 предложения) смешные утренние прогнозы "
                    "для участников чата на основе их активности. "
                    "Стиль: ироничный, немного саркастичный, но по-дружески. Только русский язык. "
                    "Опирайся на конкретные темы из сообщений — игры, вопросы, жалобы."
                )),
                HumanMessage(content=(
                    f"Участник: {username}\n\n"
                    f"{context_line}\n\n"
                    f"Напиши смешной утренний прогноз для {username}."
                )),
            ])
            logger.info(f"Daily prozharka tokens for chat {chat_id}: {response.response_metadata.get('token_usage')}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🌅 Доброе утро, {username}:\n\n{response.content}",
            )
        except Exception as error:
            logger.warning(f"Daily prozharka failed for chat {chat_id}: {error}")


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
    if not __should_respond(update):
        return

    # Direct @mentions always go through. Keyword-triggered responses
    # are rate-limited to one per chat per cooldown window.
    if not __is_bot_mentioned(update):
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
    await __send_agent_reply(update, username, update.message.text)


# --- Startup / entry point ---

async def __on_startup(application: Application) -> None:
    await init_agent()
    await wishlist.init_tables()
    await achievements.init_tables()
    await features.init_table()
    await game_filters.init_tables()
    await psstore.init_sale_tracking()

    # Check pending feature requests against current feature set and announce newly implemented ones
    newly_done = await features.find_newly_implemented()
    for chat_id, implemented_requests in newly_done.items():
        lines = [f"• {req.description}" for req in implemented_requests]
        try:
            await application.bot.send_message(
                chat_id=chat_id,
                text="🆕 Новое обновление! Следующие запросы теперь реализованы:\n\n"
                     + "\n".join(lines),
            )
        except Exception as send_error:
            logger.warning(f"Could not announce features to chat {chat_id}: {send_error}")

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
    logger.info("Bot started, all tables and jobs initialized")


def main() -> None:
    app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_TOKEN)
        .post_init(__on_startup)
        .build()
    )

    # Member tracking runs before all other handlers
    app.add_handler(TypeHandler(Update, __track_member), group=-1)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("games", cmd_games))
    app.add_handler(CommandHandler("crossplay", cmd_crossplay))
    app.add_handler(CommandHandler("players", cmd_players))
    app.add_handler(CommandHandler("chto_takoe", cmd_chto_takoe))
    app.add_handler(CommandHandler("coop", cmd_coop))
    app.add_handler(CommandHandler("play", cmd_play))
    app.add_handler(CommandHandler("wish", cmd_wish))
    app.add_handler(CommandHandler("achievements", cmd_achievements))
    app.add_handler(CommandHandler("feature", cmd_feature))
    app.add_handler(CommandHandler("features", cmd_features))
    app.add_handler(CommandHandler("rank", cmd_rank))
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("prozharka", cmd_prozharka))
    app.add_handler(CommandHandler("ban", cmd_ban))
    app.add_handler(CommandHandler("unban", cmd_unban))
    app.add_handler(CommandHandler("known", cmd_known))
    app.add_handler(CommandHandler("myfilters", cmd_myfilters))

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
            handle_message,
        )
    )

    logger.info("Starting polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
