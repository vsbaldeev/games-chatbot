import logging
import random

from telegram import Update
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown

from src import achievements, config, game_tracker
from src.agent import DailyLimitError, RateLimitError, run_agent
from src.helpers import get_username, extract_game_card, notify_unlocks

logger = logging.getLogger(__name__)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет. Я здесь. Спрашивайте про игры — если, конечно, есть что спросить."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет. Я здесь.\n\n"
        "Чтобы всё работало правильно:\n\n"
        "• Напишите что-нибудь в чат — каждый участник регистрируется, когда впервые пишет сообщение, реагирует или отправляет медиа. Только зарегистрированные попадают в рулетку и дуэли.\n"
        "Что умею:\n"
        "/dnd_pvp, /dnd_coop, /dnd_heist — D&D-приключение (PvP, кооп с боссом, великое ограбление)\n"
        "/duel — эмодзи-дуэль между двумя участниками\n"
        "/ruletka — русская рулетка\n"
        "/prozharka — прожарка случайного участника\n"
        "/multiplayer — одна кооп/онлайн игра PS5 с ценой\n"
        "/singleplayer — одна одиночная игра PS5 с ценой\n"
        "/achievements — твои достижения\n"
        "/top — топ чата\n\n"
        "Ещё отвечаю на вопросы про игры, если упомянуть меня через @.",
    )


async def __send_game_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    game_type: str,
    prompt: str,
) -> None:
    username = get_username(update)
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    await update.message.chat.send_action("typing")
    try:
        response = await run_agent(str(chat_id), username, prompt)
        card = extract_game_card(response)

        game_name = card.splitlines()[0].strip().lstrip("🎮").strip() if card else None

        await update.message.reply_text(card)

        if game_name:
            await game_tracker.mark_suggested(chat_id, game_name, game_type)

        await notify_unlocks(context, chat_id, user_id, username)
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
    chat_id = update.effective_chat.id
    username = get_username(update)
    earned = await achievements.get_user_achievements(update.effective_user.id, chat_id)
    if not earned:
        await update.message.reply_text(
            f"{username}, достижений нет. Либо ты новенький, либо слишком нормальный. Оба варианта подозрительны."
        )
        return
    safe_username = escape_markdown(username, version=1)
    recent = earned[-3:]
    lines = [
        f"{achievement.emoji} *{achievement.title}*\n_{achievement.description}_"
        for achievement in recent
    ]
    total = len(earned)
    header = f"Последние достижения {safe_username} ({total} всего):"
    await update.message.reply_text(
        f"{header}\n\n" + "\n\n".join(lines),
        parse_mode="Markdown",
    )


async def cmd_top(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    summary = await achievements.get_chat_achievements_summary(chat_id)

    if not summary:
        await update.message.reply_text("Ни у кого нет достижений. Пишите, реагируйте — зарабатывайте.")
        return

    ranked = sorted(summary.items(), key=lambda item: len(item[1]), reverse=True)[:3]

    medals = ["🥇", "🥈", "🥉"]
    lines = []
    for index, (username, earned) in enumerate(ranked):
        count = len(earned)
        if 11 <= (count % 100) <= 14:
            label = f"{count} достижений"
        elif (count % 10) == 1:
            label = f"{count} достижение"
        elif 2 <= (count % 10) <= 4:
            label = f"{count} достижения"
        else:
            label = f"{count} достижений"
        lines.append(f"{medals[index]} {username} — {label}")

    await update.message.reply_text("🏆 Топ чата:\n\n" + "\n".join(lines))
