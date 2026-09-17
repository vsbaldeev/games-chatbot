"""HTTP client for the self-hosted download service (download-service/).

Mirrors the imagegen never-raise contract: each function returns
(video_bytes, info) or None on any failure. Downloads finish in seconds,
not imagegen's minutes, so this is a plain request/response — no job/poll
machinery.
"""

import base64

import httpx

from src import config, log

logger = log.get_logger(__name__)

SHORTS_TIMEOUT_SECONDS = 90
REEL_TIMEOUT_SECONDS = 60
YOUTUBE_VIDEO_TIMEOUT_SECONDS = 30


async def download_short(url: str) -> tuple[bytes, dict] | None:
    """Download one YouTube Short via the download service.

    Args:
        url: Canonical Shorts URL.

    Returns:
        ``(video_bytes, info)`` on success, or None when the service is
        disabled (``DOWNLOAD_SERVICE_URL`` empty), unreachable, or the
        download failed — never raises.
    """
    return await post_download(url, "youtube_short", SHORTS_TIMEOUT_SECONDS)


async def download_reel(url: str) -> tuple[bytes, dict] | None:
    """Download one Instagram Reel via the download service.

    Args:
        url: Canonical Reel URL.

    Returns:
        ``(video_bytes, info)`` on success, or None on any failure.
    """
    return await post_download(url, "instagram_reel", REEL_TIMEOUT_SECONDS)


async def fetch_youtube_video(url: str) -> dict | None:
    """Fetch a long-form YouTube video's metadata via the download service.

    Args:
        url: Canonical watch URL.

    Returns:
        yt-dlp's info dict, or None on any failure.
    """
    result = await post_download(url, "youtube_video", YOUTUBE_VIDEO_TIMEOUT_SECONDS)
    return result[1] if result is not None else None


async def post_download(url: str, kind: str, timeout_seconds: float) -> tuple[bytes | None, dict] | None:
    """POST one download request and decode the response.

    Args:
        url: Canonical URL to hand the service.
        kind: One of "youtube_short", "instagram_reel", "youtube_video".
        timeout_seconds: Request timeout, sized per kind.

    Returns:
        ``(video_bytes, info)`` on success (``video_bytes`` is None for the
        metadata-only ``youtube_video`` kind), or None when the service is
        disabled, unreachable, or answers with an error.
    """
    if not config.DOWNLOAD_SERVICE_URL:
        return None
    try:
        async with httpx.AsyncClient(
            base_url=config.DOWNLOAD_SERVICE_URL, timeout=timeout_seconds
        ) as service_client:
            response = await service_client.post("/download", json={"url": url, "kind": kind})
            response.raise_for_status()
            data = response.json()
    except Exception as error:
        logger.warning("%s download failed for %s: %s", kind, url, error)
        return None
    video_bytes_base64 = data.get("video_bytes_base64")
    video_bytes = base64.b64decode(video_bytes_base64) if video_bytes_base64 else None
    return video_bytes, data.get("info") or {}
