"""
Entry point for the Telegram bot.
Run with: python -m src.bot
"""

import datetime
import logging
import random
import re

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

from src import achievements, config, features, psstore, wishlist
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
MOSCOW_TZ = datetime.timezone(datetime.timedelta(hours=3))

# Cooldown between autonomous (keyword-triggered) responses in a chat.
# Direct mentions and commands are never throttled.
AUTONOMOUS_COOLDOWN_SECONDS = 60

# Per-user cooldown on /feature to prevent token abuse.
FEATURE_COOLDOWN_SECONDS = 30
__feature_last_used: dict[int, float] = {}


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

    await update.message.chat.send_action("typing")
    try:
        response = await run_agent(chat_id, username, message_text)
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
        "/games — популярные игры для PS5 прямо сейчас\n"
        "/crossplay <игра> — есть ли кросплей\n"
        "/players <игра> — сколько людей страдает в Steam\n"
        "/research <запрос> — полный анализ игры или темы\n"
        "/coop <число> — PS5 игры с онлайн кооп на N игроков\n"
        "/play [ЧЧ:ММ] [игра] — опрос кто играет сегодня + напоминание\n"
        "/wish add|list|remove|all — вишлист игр\n"
        "/explain <термин> — объяснение технического термина\n"
        "/achievements [all] — твои достижения (или всех)\n"
        "/feature <запрос> — предложить фичу (бот проверит, нет ли её уже)\n"
        "/features — список ожидающих фич от этого чата\n\n"
        "Или просто упомяни меня или напиши про игры — я не слепой."
    )


async def cmd_games(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    username = __get_username(update)
    await __send_agent_reply(
        update,
        username,
        "Какие сейчас популярные онлайн-игры для PS5? Особенно с активным сообществом и желательно с кросплеем с PC.",
    )


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


async def cmd_research(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args) if context.args else None
    if not query:
        await update.message.reply_text("Укажи что исследовать: /research Helldivers 2")
        return
    username = __get_username(update)
    await achievements.increment_stat(
        update.effective_user.id, update.effective_chat.id, username, "research_queries"
    )
    await __send_agent_reply(
        update,
        username,
        f"Сделай полный анализ: {query}. Расскажи об игре, платформах, мультиплеере, кросплее и количестве игроков.",
    )


async def cmd_explain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    term = " ".join(context.args) if context.args else None
    if not term:
        await update.message.reply_text("Что объяснить? Например: /explain ray tracing")
        return
    username = __get_username(update)
    await achievements.increment_stat(
        update.effective_user.id, update.effective_chat.id, username, "explain_queries"
    )
    await __send_agent_reply(
        update,
        username,
        f"Объясни простым языком для неспециалиста что такое: {term}",
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

    poll_question = (
        f"Кто играет сегодня в {game_name}?" if game_name else "Кто играет сегодня вечером?"
    )
    await update.message.reply_poll(
        question=poll_question,
        options=["Я в деле 🎮", "Может быть 🤔", "Не смогу 😢"],
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


async def __daily_roast(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = await achievements.get_all_chat_ids()
    if not chat_ids:
        return

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=config.GROQ_API_KEY,
        temperature=0.9,
        max_tokens=200,
    )

    for chat_id in chat_ids:
        members = await achievements.get_chat_members(chat_id)
        if not members:
            continue
        user_id, username = random.choice(members)

        history = get_chat_history(str(chat_id))
        recent = await get_recent_messages(history, 30)

        # Only use messages from the roasted user to avoid leaking other people's content.
        user_prefix = f"[{username}]:"
        user_messages = [
            msg.content for msg in recent
            if hasattr(msg, "content")
            and isinstance(msg.content, str)
            and msg.content.startswith(user_prefix)
        ]
        history_text = (
            "\n".join(user_messages)
            if user_messages
            else f"{username} ещё не проявил себя в чате — пишем гороскоп вслепую."
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(content=(
                    "Ты луркморский оракул. Пишешь короткие (2-3 предложения) утренние псевдо-гороскопы "
                    "для геймеров на основе их реальной активности в чате. "
                    "Стиль: циничный, пессимистичный, смешной. Только русский язык. "
                    "Опирайся на конкретные темы из сообщений пользователя — игры, вопросы, жалобы."
                )),
                HumanMessage(content=(
                    f"Участник: {username}\n\n"
                    f"Последние сообщения {username}:\n{history_text}\n\n"
                    f"Напиши утренний гороскоп для {username}."
                )),
            ])
            logger.info(f"Daily roast tokens for chat {chat_id}: {response.response_metadata.get('token_usage')}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🌅 Утренний прогноз для {username}:\n\n{response.content}",
            )
        except Exception as error:
            logger.warning(f"Daily roast failed for chat {chat_id}: {error}")


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
    app.add_handler(CommandHandler("research", cmd_research))
    app.add_handler(CommandHandler("coop", cmd_coop))
    app.add_handler(CommandHandler("play", cmd_play))
    app.add_handler(CommandHandler("wish", cmd_wish))
    app.add_handler(CommandHandler("explain", cmd_explain))
    app.add_handler(CommandHandler("achievements", cmd_achievements))
    app.add_handler(CommandHandler("feature", cmd_feature))
    app.add_handler(CommandHandler("features", cmd_features))

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
