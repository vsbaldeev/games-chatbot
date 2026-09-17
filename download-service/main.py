"""FastAPI app for the self-hosted yt-dlp download service.

Owns every yt-dlp download for the bot (YouTube Shorts, Instagram Reels,
long-form YouTube metadata) so a yt-dlp version bump restarts only this
container, never the bot. See README.md for the full contract.
"""

import asyncio
import contextlib
import datetime
import logging

from fastapi import FastAPI

from ytdlp_update import check_and_update

logger = logging.getLogger("download-service")

UPDATE_CHECK_HOUR_UTC = 3
UPDATE_CHECK_MINUTE_UTC = 30


def seconds_until_next_check(now: datetime.datetime | None = None) -> float:
    """Seconds from ``now`` until the next 03:30 UTC daily check.

    Args:
        now: Current UTC time; defaults to ``datetime.datetime.now(UTC)``.

    Returns:
        Seconds until the next occurrence of the daily check time, always
        in ``(0, 24*3600]``.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    target = now.replace(
        hour=UPDATE_CHECK_HOUR_UTC, minute=UPDATE_CHECK_MINUTE_UTC, second=0, microsecond=0
    )
    if target <= now:
        target += datetime.timedelta(days=1)
    return (target - now).total_seconds()


async def daily_update_loop() -> None:
    """Run check_and_update once a day at UPDATE_CHECK_HOUR_UTC:MINUTE_UTC."""
    while True:
        await asyncio.sleep(seconds_until_next_check())
        await check_and_update()


@contextlib.asynccontextmanager
async def lifespan(application: FastAPI):
    """Run the daily update loop for the app's lifetime.

    Args:
        application: The FastAPI application being started.

    Yields:
        Control to the running application.
    """
    updater = asyncio.get_running_loop().create_task(daily_update_loop())
    yield
    updater.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    """Report service liveness.

    Returns:
        Status payload.
    """
    return {"status": "ok"}
