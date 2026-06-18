"""Scheduled job: sweep silent users for silence-based achievements."""

from src import achievements, log
from telegram.error import BadRequest
from telegram.ext import ContextTypes

logger = log.get_logger(__name__)


async def silence_sweep_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = await achievements.get_all_chat_ids()
    for chat_id in chat_ids:
        members = await achievements.get_chat_members(chat_id)
        for user_id, username in members:
            try:
                new_ones = await achievements.check_silence_achievements(user_id, chat_id, username)
                for ach in new_ones:
                    text = (
                        f"🏆 @{username} получил достижение!\n\n"
                        f"{ach.emoji} <b>{ach.title}</b>\n<i>{ach.description}</i>"
                    )
                    try:
                        await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
                    except BadRequest:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=f"🏆 {username}: {ach.emoji} {ach.title} — {ach.description}",
                        )
            except Exception as error:
                logger.warning("Silence check failed for user %s in chat %s: %s", user_id, chat_id, error)
