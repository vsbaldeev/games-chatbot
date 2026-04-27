import logging
import random
import re

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, MessageEntity, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes, Job

from src import achievements, config
from src.helpers import notify_unlocks

logger = logging.getLogger(__name__)

DUEL_CALLBACK_DATA = "duel_fire"
DUEL_CHALLENGE_RE = re.compile(r"дуэл", re.IGNORECASE)

__MD_SPECIAL_RE = re.compile(r"([_*`\[])")


def __escape_md(text: str) -> str:
    return __MD_SPECIAL_RE.sub(r"\\\1", text)


def __fmt(template: str, **kwargs: str) -> str:
    """Safe substitution that handles usernames containing { or }."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", value)
    return result


__DUEL_ANNOUNCE = [
    "⚔️ Эмодзи-дуэль!\n\n@{p1} против @{p2}\n\nПервый нажмёт на курок — победит. Приготовьтесь...",
    "🔫 Дуэль объявляется!\n\n@{p1} vs @{p2}\n\nОдна кнопка. Один победитель. Удачи.",
    "⚔️ Судьба свела двоих.\n\n@{p1} и @{p2} — к барьеру!\n\nКто нажмёт первым?",
    "🎯 Внимание, чат! Дуэль!\n\n@{p1} против @{p2}\n\nПервый нажмёт — тот и прав.",
    "💀 Двое вошли. Один выйдет.\n\n@{p1} vs @{p2}\n\nКурок ждёт.",
]

__DUEL_WIN = [
    "💥 *@{winner}* выстрелил первым!\n\n@{loser} — слишком медленно. В следующий раз.",
    "🔫 *@{winner}* нажал! БАХ! 💀\n\n@{loser} опоздал на долю секунды.",
    "⚡ *@{winner}* — реакция молнии! 🏆\n\n@{loser} даже не успел понять, что произошло.",
    "🎯 Точно в цель! *@{winner}* побеждает!\n\n@{loser} — медленнее черепахи. Позор.",
    "🤠 *@{winner}* быстрее! Дуэль окончена.\n\n@{loser} уходит ни с чем.",
]

__DUEL_EXPIRED = [
    "⏳ Время вышло. @{p1} и @{p2} так и не решились нажать. Трусы.",
    "💨 Дуэль @{p1} vs @{p2} отменяется — оба испугались курка.",
    "🏳️ Никто не нажал. @{p1} и @{p2} разошлись по домам, поджав хвосты.",
]

DUEL_TIMEOUT_SECONDS = 300

# message_id → (chat_id, p1_id, p1_username, p2_id, p2_username)
__pending_duels: dict[int, tuple[int, int, str, int, str]] = {}
# message_id → job (for cancellation when duel resolves early)
__duel_jobs: dict[int, Job] = {}
# chat_ids with an active duel (at most one per chat)
__active_duel_chats: set[int] = set()


async def __send_duel(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    p1_id: int,
    p1_username: str,
    p2_id: int,
    p2_username: str,
    reply_to_message_id: int | None = None,
) -> bool:
    """Start a duel. Returns False if there is already an active duel in this chat."""
    if chat_id in __active_duel_chats:
        return False
    __active_duel_chats.add(chat_id)
    announce = __fmt(random.choice(__DUEL_ANNOUNCE), p1=p1_username, p2=p2_username)
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🔫", callback_data=DUEL_CALLBACK_DATA)]])
    msg = await context.bot.send_message(
        chat_id=chat_id,
        text=announce,
        reply_markup=keyboard,
        reply_to_message_id=reply_to_message_id,
    )
    __pending_duels[msg.message_id] = (chat_id, p1_id, p1_username, p2_id, p2_username)
    job = context.job_queue.run_once(__expire_duel, DUEL_TIMEOUT_SECONDS, data=msg.message_id)
    __duel_jobs[msg.message_id] = job
    return True


async def cmd_duel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    members = await achievements.get_chat_members(chat_id)
    if len(members) < 2:
        await update.message.reply_text("Недостаточно участников для дуэли. Нужно хотя бы 2.")
        return

    caller_id = update.effective_user.id
    eligible = [member for member in members if member[0] != caller_id]
    pool = eligible if len(eligible) >= 2 else members
    (p1_id, p1_username), (p2_id, p2_username) = random.sample(pool, 2)

    started = await __send_duel(
        context, chat_id, p1_id, p1_username, p2_id, p2_username,
        reply_to_message_id=update.message.message_id,
    )
    if not started:
        await update.message.reply_text("В чате уже идёт дуэль. Дождитесь её завершения.")


async def handle_duel_mention(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Start a targeted duel if the message contains a valid non-bot @mention. Returns True if a duel was started."""
    bot_username = config.BOT_USERNAME.lower().lstrip("@")
    text = update.message.text or ""

    target_username = None
    for entity in (update.message.entities or []):
        if entity.type != MessageEntity.MENTION:
            continue
        mentioned = text[entity.offset + 1: entity.offset + entity.length]
        if mentioned.lower() != bot_username:
            target_username = mentioned
            break

    if not target_username:
        return False

    chat_id = update.effective_chat.id
    challenger_id = update.effective_user.id
    challenger_username = (
        update.effective_user.username
        or update.effective_user.first_name
        or f"user_{challenger_id}"
    )

    members = await achievements.get_chat_members(chat_id)
    target = next(
        ((uid, uname) for uid, uname in members if uname.lower() == target_username.lower()),
        None,
    )

    if not target:
        await update.message.reply_text(f"Не нашёл @{target_username} среди участников чата.")
        return True

    target_id, target_uname = target
    if target_id == challenger_id:
        await update.message.reply_text("Нельзя вызвать на дуэль самого себя.")
        return True

    started = await __send_duel(
        context, chat_id,
        challenger_id, challenger_username,
        target_id, target_uname,
        reply_to_message_id=update.message.message_id,
    )
    if not started:
        await update.message.reply_text("В чате уже идёт дуэль. Дождитесь её завершения.")
    return True


async def handle_duel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    message_id = query.message.message_id
    clicker_id = query.from_user.id

    # Pop atomically before any await — asyncio is single-threaded so this is safe
    claimed = __pending_duels.pop(message_id, None)
    if not claimed:
        await query.answer("Дуэль уже завершена.", show_alert=True)
        return

    chat_id, p1_id, p1_username, p2_id, p2_username = claimed

    if clicker_id not in (p1_id, p2_id):
        # Re-insert so the duel remains open for the actual participants
        __pending_duels[message_id] = claimed
        await query.answer("Это не твоя дуэль, зритель! 👀", show_alert=True)
        return

    job = __duel_jobs.pop(message_id, None)
    if job:
        job.schedule_removal()

    winner_username = query.from_user.username or query.from_user.first_name or f"user_{clicker_id}"
    loser_username = p2_username if clicker_id == p1_id else p1_username

    await query.answer("💥 БАХ! Ты выстрелил первым!")

    win_template = random.choice(__DUEL_WIN)
    result_text = __fmt(
        win_template,
        winner=__escape_md(winner_username),
        loser=__escape_md(loser_username),
    )
    try:
        await query.edit_message_text(result_text, parse_mode="Markdown")
    except BadRequest:
        await query.edit_message_text(
            __fmt(win_template, winner=winner_username, loser=loser_username)
        )

    __active_duel_chats.discard(chat_id)
    await achievements.increment_stat(clicker_id, chat_id, winner_username, "duel_wins")
    await notify_unlocks(context, chat_id, clicker_id, winner_username)


async def __expire_duel(context: ContextTypes.DEFAULT_TYPE) -> None:
    message_id = context.job.data
    duel = __pending_duels.pop(message_id, None)
    __duel_jobs.pop(message_id, None)
    if not duel:
        return

    chat_id, p1_id, p1_username, p2_id, p2_username = duel
    __active_duel_chats.discard(chat_id)
    text = __fmt(random.choice(__DUEL_EXPIRED), p1=p1_username, p2=p2_username)
    try:
        await context.bot.edit_message_reply_markup(
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=None,
        )
        await context.bot.send_message(chat_id=chat_id, text=text)
    except TelegramError as error:
        logger.warning(f"Duel expiry failed for message {message_id}: {error}")
