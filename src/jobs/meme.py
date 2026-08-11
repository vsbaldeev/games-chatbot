"""Scheduled job: send one fresh meme to every known chat, once per day."""

import asyncio

from telegram.ext import ContextTypes

from src import achievements, log
from src.memes.sender import send_meme

logger = log.get_logger(__name__)


async def daily_meme_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a daily meme to every chat that has registered members.

    The meme is un-anchored — it opens the day rather than answering anyone.
    Failures are absorbed per chat by ``send_meme`` and by
    ``return_exceptions``, so one dead chat never costs the others their meme.
    """
    chat_ids = await achievements.get_all_chat_ids()
    await asyncio.gather(
        *[send_meme(context.bot, chat_id) for chat_id in chat_ids],
        return_exceptions=True,
    )
