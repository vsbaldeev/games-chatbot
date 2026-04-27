import logging

from telegram.error import BadRequest
from telegram.ext import ContextTypes

from src import achievements

logger = logging.getLogger(__name__)


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
                logger.warning(f"Silence achievement check failed for user {user_id} in chat {chat_id}: {error}")
