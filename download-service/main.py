"""FastAPI app for the self-hosted yt-dlp download service.

Owns every yt-dlp download for the bot (YouTube Shorts, Instagram Reels,
long-form YouTube metadata) so a yt-dlp version bump restarts only this
container, never the bot. See README.md for the full contract.
"""

import asyncio
import base64
import contextlib
import datetime
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import instagram_reel
import shorts
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


class DownloadRequest(BaseModel):
    """Body of ``POST /download``.

    Attributes:
        url: Canonical URL to download or fetch metadata for.
        kind: Which relocated module handles the request.
    """

    url: str
    kind: Literal["youtube_short", "instagram_reel"]


class DownloadResponse(BaseModel):
    """Body of a successful ``POST /download`` response.

    Attributes:
        video_bytes_base64: Base64-encoded video bytes, or None for a
            metadata-only kind.
        info: yt-dlp's info dict for the downloaded/fetched item.
    """

    video_bytes_base64: str | None
    info: dict


# (module, attribute name) rather than a bound function reference: resolved
# via getattr at call time, so patching e.g. "main.shorts.download_short" in
# tests reaches the actual call instead of a reference captured at import.
DOWNLOADERS = {
    "youtube_short": (shorts, "download_short"),
    "instagram_reel": (instagram_reel, "download_reel"),
}


@app.post("/download", response_model=DownloadResponse)
def download(request: DownloadRequest) -> DownloadResponse:
    """Download or fetch metadata for one URL via the matching relocated module.

    Args:
        request: The URL and which downloader to use.

    Returns:
        Base64-encoded video bytes (None for metadata-only kinds) + the
        info dict.

    Raises:
        HTTPException: 502 when the downloader raised (extraction failure,
            timeout, duration/filesize rejection, ...).
    """
    module, attr_name = DOWNLOADERS[request.kind]
    downloader = getattr(module, attr_name)
    try:
        video_bytes, info = downloader(request.url)
    except Exception as error:
        logger.warning("download failed for %s (%s): %s", request.url, request.kind, error)
        raise HTTPException(status_code=502, detail=str(error)) from error
    video_bytes_base64 = base64.b64encode(video_bytes).decode() if video_bytes is not None else None
    return DownloadResponse(video_bytes_base64=video_bytes_base64, info=info)
