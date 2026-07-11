"""Scheduled job: silently refresh Жора's current activity once a day.

Runs at :data:`DAILY_ACTIVITY_RUN_TIME` (09:30 Moscow Time), before the
life-post window opens at 10:00 (``src/jobs/life_post.py``), so a life post
scheduled for later today always ends up as the newer ``current_activity``
row. Posts nothing to chat — see ``src/life/activity.py`` for the generation
flow.
"""

import datetime

from telegram.ext import ContextTypes

from src import log
from src.jobs.life_post import LIFE_POST_TIMEZONE
from src.life.activity import refresh_daily_activity
from src.store import bot_memories

logger = log.get_logger(__name__)

DAILY_ACTIVITY_RUN_TIME = datetime.time(hour=9, minute=30, tzinfo=LIFE_POST_TIMEZONE)
CATCH_UP_DELAY_SECONDS = 60


def refreshed_today(posted_at: float, now: datetime.datetime) -> bool:
    """Return True when ``posted_at`` falls on the same Moscow calendar date as ``now``.

    Args:
        posted_at: Unix timestamp of the newest stored activity.
        now: Current timezone-aware moment in Moscow Time.

    Returns:
        True when both timestamps share the same Moscow calendar date.
    """
    posted = datetime.datetime.fromtimestamp(posted_at, tz=now.tzinfo)
    return posted.date() == now.date()


async def daily_activity_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Daily 09:30 MSK callback: refresh today's activity unless one already exists.

    A life post that already landed today also counts (it wrote a newer
    ``current_activity`` than anything the refresh could produce), so the
    check is a plain "already refreshed today", not "was it an activity row".
    """
    now = datetime.datetime.now(LIFE_POST_TIMEZONE)
    activity = await bot_memories.get_current_activity()
    if activity is not None and refreshed_today(activity[1], now):
        logger.info("Current activity already refreshed today — skipping")
        return
    await refresh_daily_activity()


async def catch_up_daily_activity_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """On startup, recover a daily refresh missed while the bot was down.

    Refreshes immediately when there is no activity at all, or the newest
    one predates today (Moscow) and the daily run time has already passed;
    otherwise the regular :func:`daily_activity_job` will run later today.
    """
    activity = await bot_memories.get_current_activity()
    if activity is None:
        logger.info("No current activity has ever been recorded — refreshing now")
        await refresh_daily_activity()
        return

    now = datetime.datetime.now(LIFE_POST_TIMEZONE)
    if refreshed_today(activity[1], now):
        logger.info("Current activity already refreshed today; skipping startup catch-up")
        return
    if now.time() >= DAILY_ACTIVITY_RUN_TIME.replace(tzinfo=None):
        logger.info("Missed today's daily activity refresh — running startup catch-up")
        await refresh_daily_activity()
        return
    logger.info("Daily activity run time hasn't passed yet — the scheduled job will handle it")
