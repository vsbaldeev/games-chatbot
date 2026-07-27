"""Scheduled job: post an episode from Жора's life on a fixed weekly schedule.

Three posts a week, each at 17:00 Moscow Time, each in the format this
schedule assigns:

===========  =======  ==================================================
Monday       photo    generated frame, ``episode_text`` as its caption
Wednesday    voice    spoken story, ``voice_teaser`` as its caption
Saturday     story    plain text
===========  =======  ==================================================

The format is a scheduling decision, not a creative one. It used to be the
episode writer's free choice among the offered formats, constrained only by
"don't repeat the previous post" — a rule that endless story/voice
alternation satisfies forever, so ``photo`` was never once picked in
production. Assigning the format here guarantees each one gets its slot.

The very first post after a fresh deployment fires right away; a slot the
bot was down for is recovered on startup.
"""

import datetime
from zoneinfo import ZoneInfo

from telegram.ext import ContextTypes

from src import log
from src.life.poster import post_life_episode
from src.life.writer import PHOTO_FORMAT, STORY_FORMAT, VOICE_FORMAT
from src.store import bot_memories

logger = log.get_logger(__name__)

LIFE_POST_TIMEZONE = ZoneInfo("Europe/Moscow")

# Keyed by datetime.weekday() — Monday is 0. Days absent from the map post
# nothing, so the number of posts per week is just this table's size.
WEEKLY_SCHEDULE: dict[int, str] = {
    0: PHOTO_FORMAT,
    2: VOICE_FORMAT,
    5: STORY_FORMAT,
}

# Moscow Time is a fixed UTC+3 offset (no DST), but APScheduler resolves this
# tzinfo correctly either way (see JobQueue.run_daily), so the job reliably
# fires at 17:00 local time year-round.
LIFE_POST_RUN_TIME = datetime.time(hour=17, minute=0, tzinfo=LIFE_POST_TIMEZONE)

# Catch-up only: the scheduled slot is always 17:00, but a bot that starts at
# 04:00 owing a missed post must not wake the chat to deliver it.
LIFE_POST_WINDOW = (10, 22)  # local hours [start, end)

# A fresh deployment's opener introduces Жора to the chat, so it is plain
# text: there is no canon yet for a photo to depict or a voice note to tease.
OPENER_FORMAT = STORY_FORMAT

CATCH_UP_DELAY_SECONDS = 60


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


def most_recent_due_slot(now: datetime.datetime) -> tuple[datetime.datetime, str] | None:
    """Return the latest scheduled slot at or before ``now``, with its format.

    Args:
        now: Current timezone-aware moment in Moscow Time.

    Returns:
        ``(moment, post_format)`` for the most recent slot already due, or
        None when :data:`WEEKLY_SCHEDULE` is empty. Looks back 8 days so a
        today-but-not-yet-17:00 slot still resolves to the previous one.
    """
    for days_back in range(8):
        day = now - datetime.timedelta(days=days_back)
        post_format = WEEKLY_SCHEDULE.get(day.weekday())
        if post_format is None:
            continue
        slot = day.replace(
            hour=LIFE_POST_RUN_TIME.hour, minute=LIFE_POST_RUN_TIME.minute,
            second=0, microsecond=0,
        )
        if slot <= now:
            return slot, post_format
    return None


async def run_post_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """JobQueue callback that actually sends a deferred post.

    The format travels in ``job.data`` — the slot it was deferred from
    decides it, not the moment it finally runs.
    """
    await post_life_episode(context.bot, context.job.data)


async def schedule_deferred_post(context: ContextTypes.DEFAULT_TYPE, post_format: str) -> None:
    """Post now if inside the daytime window, otherwise defer to the next window start.

    Args:
        context: Job context supplying the queue to schedule on.
        post_format: Format the deferred post must use.
    """
    now = datetime.datetime.now(LIFE_POST_TIMEZONE)
    target = next_window_start(now)
    delay_seconds = max(0, (target - now).total_seconds())
    context.job_queue.run_once(run_post_job, when=delay_seconds, data=post_format)


async def life_post_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Post today's episode when today is a scheduled day.

    Registered to run daily at :data:`LIFE_POST_RUN_TIME`; days missing from
    :data:`WEEKLY_SCHEDULE` return without posting.
    """
    now = datetime.datetime.now(LIFE_POST_TIMEZONE)
    post_format = WEEKLY_SCHEDULE.get(now.weekday())
    if post_format is None:
        return
    logger.info("Scheduled life post due — format %s", post_format)
    await post_life_episode(context.bot, post_format)


async def catch_up_life_post_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """On startup, post the deployment opener or recover a missed scheduled slot.

    When no episode has ever been posted, this is a fresh deployment: the
    very first life post fires now (deferred to the next daytime window if
    started at night). Otherwise, recovers a scheduled slot the bot was down
    for, in that slot's own format.
    """
    latest_posted_at = await bot_memories.get_latest_posted_at()
    if latest_posted_at is None:
        logger.info("No life post has ever been sent — posting the deployment opener")
        await schedule_deferred_post(context, OPENER_FORMAT)
        return

    now = datetime.datetime.now(LIFE_POST_TIMEZONE)
    due = most_recent_due_slot(now)
    if due is not None and latest_posted_at < due[0].timestamp():
        last_due, post_format = due
        logger.info(
            "Missed scheduled life post (%s, format %s) — running startup catch-up",
            last_due, post_format,
        )
        await schedule_deferred_post(context, post_format)
        return
    logger.info("Life posts up to date; skipping startup catch-up")
