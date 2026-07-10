"""Scheduled job: post a life-story episode from Жора's life, twice a week.

Posts land at random daytime moments in Moscow Time — never at
night, so a proactive post never lands while chat members are asleep (the
reactive pipeline still answers mentions and replies around the clock; this
job only governs proactive posting). The very first post ever fires right
after deployment; after that, a deterministic per-week random plan decides
which two days post, mirroring the seeded-plan pattern used elsewhere for
weekly variety without a schedule table.
"""

import datetime
import random
from zoneinfo import ZoneInfo

from telegram.ext import ContextTypes

from src import log
from src.life.poster import post_life_episode
from src.store import bot_memories

logger = log.get_logger(__name__)

LIFE_POST_TIMEZONE = ZoneInfo("Europe/Moscow")
LIFE_POST_WINDOW = (10, 22)  # local hours [start, end) — no night posts
LIFE_POSTS_PER_WEEK = 2

# Moscow Time is a fixed UTC+3 offset (no DST), but APScheduler resolves this
# tzinfo correctly either way (see JobQueue.run_daily), so the job reliably
# fires at 10:00 local time year-round.
LIFE_POST_RUN_TIME = datetime.time(hour=LIFE_POST_WINDOW[0], minute=0, tzinfo=LIFE_POST_TIMEZONE)

CATCH_UP_DELAY_SECONDS = 60


def week_plan(now: datetime.datetime) -> list[datetime.datetime]:
    """Return this ISO week's planned post moments, deterministic for the week.

    Args:
        now: A timezone-aware moment in Moscow Time; only its ISO
            year/week identify the plan, so any moment during the week
            returns the same result.

    Returns:
        ``LIFE_POSTS_PER_WEEK`` timezone-aware moments, chronologically
        sorted, each a random minute inside :data:`LIFE_POST_WINDOW` on a
        random day of that ISO week.
    """
    iso_year, iso_week, _ = now.isocalendar()
    rng = random.Random(f"life-{iso_year}-{iso_week}")
    days = rng.sample(range(7), LIFE_POSTS_PER_WEEK)
    start_hour, end_hour = LIFE_POST_WINDOW
    window_minutes = (end_hour - start_hour) * 60
    week_start = (now - datetime.timedelta(days=now.isoweekday() - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    moments = [
        week_start
        + datetime.timedelta(days=day, hours=start_hour, minutes=rng.randrange(window_minutes))
        for day in days
    ]
    return sorted(moments)


def next_window_start(now: datetime.datetime) -> datetime.datetime:
    """Return ``now`` if inside the daytime window, otherwise the next window start.

    Args:
        now: Current timezone-aware moment in Moscow Time.

    Returns:
        ``now`` unchanged when inside :data:`LIFE_POST_WINDOW`; otherwise
        today's window start if still ahead, tomorrow's otherwise — this is
        the night-deferral rule applied to catch-up posts.
    """
    start_hour, end_hour = LIFE_POST_WINDOW
    if start_hour <= now.hour < end_hour:
        return now
    today_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    return today_start if now.hour < start_hour else today_start + datetime.timedelta(days=1)


def most_recent_due_slot(now: datetime.datetime) -> datetime.datetime | None:
    """Return the latest planned slot at or before ``now``.

    Checks both this and the previous ISO week's plan, since a slot near a
    week boundary can belong to either.

    Args:
        now: Current timezone-aware moment in Moscow Time.

    Returns:
        The latest planned moment at/before ``now``, or None if every
        candidate slot is still in the future.
    """
    candidates = week_plan(now) + week_plan(now - datetime.timedelta(days=7))
    due = [moment for moment in candidates if moment <= now]
    return max(due) if due else None


async def run_post_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """JobQueue callback that actually sends the scheduled post."""
    await post_life_episode(context.bot)


async def schedule_deferred_post(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Post now if inside the daytime window, otherwise defer to the next window start."""
    now = datetime.datetime.now(LIFE_POST_TIMEZONE)
    target = next_window_start(now)
    delay_seconds = max(0, (target - now).total_seconds())
    context.job_queue.run_once(run_post_job, when=delay_seconds)


async def life_post_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Post today's scheduled episode, if today is one of this week's planned days.

    Registered to run daily at the window start (:data:`LIFE_POST_RUN_TIME`);
    schedules a one-off run at the planned minute so the post lands at an
    organic-feeling time rather than exactly on the hour.
    """
    now = datetime.datetime.now(LIFE_POST_TIMEZONE)
    for slot in week_plan(now):
        if slot.date() == now.date():
            delay_seconds = max(0, (slot - now).total_seconds())
            context.job_queue.run_once(run_post_job, when=delay_seconds)


async def catch_up_life_post_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """On startup, post the deployment opener or recover a missed scheduled slot.

    When no episode has ever been posted, this is a fresh deployment: the
    very first life post fires now (deferred to the next daytime window if
    started at night). Otherwise, recovers a scheduled slot the bot was down
    for, using the same night-deferral rule.
    """
    latest_posted_at = await bot_memories.get_latest_posted_at()
    if latest_posted_at is None:
        logger.info("No life post has ever been sent — posting the deployment opener")
        await schedule_deferred_post(context)
        return

    now = datetime.datetime.now(LIFE_POST_TIMEZONE)
    last_due = most_recent_due_slot(now)
    if last_due is not None and latest_posted_at < last_due.timestamp():
        logger.info("Missed scheduled life post (%s) — running startup catch-up", last_due)
        await schedule_deferred_post(context)
        return
    logger.info("Life posts up to date; skipping startup catch-up")
