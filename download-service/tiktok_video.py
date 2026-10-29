"""Anonymous yt-dlp download logic for TikTok videos.

Best-effort, no authentication, same philosophy as instagram_reel.py: any
failure (blocked, private, deleted, timeout, a photo slideshow with no video
stream) is raised to the caller (main.py), which turns it into a 502 — the
bot-side client is where this degrades to None. No retry signal yet: unlike
Instagram's access gate, no TikTok failure has been observed to be flaky
rather than hard, so every failure fails fast.

Dedup gates and the daily cap stay in the bot — none of that is
yt-dlp-specific.
"""

import os
import tempfile

import yt_dlp

MAX_FILESIZE_BYTES = 50 * 1024 * 1024  # Telegram Bot API upload cap
SOCKET_TIMEOUT_SECONDS = 20

# TikTok serves both H.264 and H.265 (bytevc1) renditions; Telegram clients
# play H.265 unreliably, so H.264 is preferred and anything else is only the
# fallback when no H.264 rendition fits under the cap.
TIKTOK_FORMAT = (
    f"b[vcodec^=h264][filesize<{MAX_FILESIZE_BYTES}]"
    f"/b[vcodec^=avc][filesize<{MAX_FILESIZE_BYTES}]"
    f"/b[filesize<{MAX_FILESIZE_BYTES}]/b"
)


def build_ydl_opts(target_dir: str) -> dict:
    """Assemble yt-dlp options for an anonymous, filesize-capped download.

    Args:
        target_dir: Directory the downloaded file is written into.

    Returns:
        Options dict for ``yt_dlp.YoutubeDL``.
    """
    return {
        "format": TIKTOK_FORMAT,
        "outtmpl": os.path.join(target_dir, "tiktok.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": SOCKET_TIMEOUT_SECONDS,
        "max_filesize": MAX_FILESIZE_BYTES,
    }


def download_tiktok(url: str) -> tuple[bytes, dict]:
    """Download one TikTok video into a temporary directory (blocking).

    Args:
        url: Canonical TikTok video URL or a vm./vt./t/ short link.

    Returns:
        Tuple of the downloaded video bytes and yt-dlp's info dict.

    Raises:
        yt_dlp.utils.DownloadError: On extraction or download failure.
        FileNotFoundError: When the filesize guard rejected the video or
            nothing was downloaded.
    """
    with tempfile.TemporaryDirectory() as target_dir:
        with yt_dlp.YoutubeDL(build_ydl_opts(target_dir)) as ydl:
            info = ydl.extract_info(url, download=True)
        requested = (info or {}).get("requested_downloads") or []
        filepath = requested[0].get("filepath") if requested else None
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError(f"TikTok rejected by filesize guard or not downloaded: {url}")
        with open(filepath, "rb") as video_file:
            return video_file.read(), info
