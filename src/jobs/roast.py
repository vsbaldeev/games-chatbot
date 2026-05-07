"""Scheduled job: weekly roast — fires once per week on a deterministic random day."""

import asyncio
import datetime
import random

from telegram.ext import ContextTypes

from src import achievements, log
from src.commands.fun.roast import generate_roast_text

logger = log.get_logger(__name__)


def _is_roast_day() -> bool:
    """Return True if today is this week's roast day.

    The day is derived deterministically from the ISO year+week so it is
    stable across restarts yet varies week to week without any stored state.
    """
    today = datetime.date.today()
    year, week, _ = today.isocalendar()
    rng = random.Random(year * 1000 + week)
    return today.weekday() == rng.randint(0, 6)


async def _roast_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    members = await achievements.get_chat_members(chat_id)
    if not members:
        return
    target_id, target_username = random.choice(members)
    try:
        header, roast_text = await generate_roast_text(chat_id, target_id, target_username)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"{header} Еженедельная #прожарка @{target_username}\n\n{roast_text}",
        )
        await achievements.increment_stat(target_id, chat_id, target_username, "roasted_count")
    except Exception as error:
        logger.warning("Weekly roast failed for chat %s: %s", chat_id, error)


async def weekly_roast_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_roast_day():
        return
    chat_ids = await achievements.get_all_chat_ids()
    await asyncio.gather(
        *[_roast_chat(context, chat_id) for chat_id in chat_ids],
        return_exceptions=True,
    )
