import datetime
import logging
import re

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from src import achievements, config

logger = logging.getLogger(__name__)

OFFENSE_RE = re.compile(
    r"(тупой|тупая|тупит|идиот|дебил|мудак|г[ао]вн[оа]|хуйн[яе]|нахуй|пиздец|"
    r"отстой|бесполезн|сломан|не работает|глупый|глупая|дерьм[оа]|придур|долбо|"
    r"ёбан|еба[нл]|заткн|иди нах|иди в|stupid|useless|broken|dumb|trash|"
    r"garbage|sucks|piece of shit|fuck)",
    re.IGNORECASE | re.UNICODE,
)

MOSCOW_TZ = datetime.timezone(datetime.timedelta(hours=3))

__TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
__INTERMEDIATE_LINE_RE = re.compile(r"^(#{1,3} |\d+\. | {2,}- )")


def fallback_username(user_id: int) -> str:
    return f"user_{user_id}"


def get_username(update: Update) -> str:
    user = update.effective_user
    return user.username or user.first_name or fallback_username(user.id)


def to_telegram_md(text: str) -> str:
    """Sanitise LLM output for Telegram Markdown v1."""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)
    lines = [line for line in text.splitlines() if not __TABLE_SEPARATOR_RE.match(line)]
    return "\n".join(lines)


def extract_game_card(response: str) -> str:
    """Strip intermediate tool-result summaries, returning only the final game card."""
    lines = response.splitlines()

    store_idx = next(
        (idx for idx in range(len(lines) - 1, -1, -1)
         if "🛒" in lines[idx] or "store.playstation.com" in lines[idx]),
        None,
    )
    if store_idx is None:
        return response

    start_idx = store_idx
    for idx in range(store_idx - 1, -1, -1):
        if __INTERMEDIATE_LINE_RE.match(lines[idx]):
            break
        start_idx = idx

    result = lines[start_idx : store_idx + 1]
    while result and not result[0].strip():
        result = result[1:]

    return "\n".join(result).strip()


def is_bot_mentioned(update: Update) -> bool:
    text = update.message.text or ""
    return config.BOT_USERNAME.lower() in text.lower()


def is_reply_to_bot(update: Update, bot_id: int) -> bool:
    reply = update.message.reply_to_message
    if not reply or not reply.from_user:
        return False
    return reply.from_user.id == bot_id


def is_reply_to_dnd_message(update: Update) -> bool:
    reply = update.message.reply_to_message
    if not reply:
        return False
    return (reply.text or "").startswith("⚔️")


def is_night_message(update: Update) -> bool:
    """True if the message was sent between 00:00 and 05:00 Moscow time."""
    if not update.message or not update.message.date:
        return False
    moscow_time = update.message.date.astimezone(MOSCOW_TZ)
    return 0 <= moscow_time.hour < 5


async def notify_unlocks(
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
    except Exception as error:
        logger.warning(f"Achievement notification failed for user {user_id} in chat {chat_id}: {error}")
