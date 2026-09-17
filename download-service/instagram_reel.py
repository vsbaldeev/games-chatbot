"""Anonymous yt-dlp download logic for Instagram Reels (relocated from
src/pipeline/social_links/instagram_reel.py).

Best-effort, no authentication: Instagram aggressively blocks anonymous
scraping, so any failure (blocked, private, deleted, timeout) is raised to
the caller (main.py), which turns it into a 502 — the bot-side client is
where this degrades to None. Instagram's access check is flaky rather than
a hard per-post block (observed: different anonymous requests to the same
network get through inconsistently), so that specific failure signature
gets a few bounded retries; every other failure (private, deleted,
unsupported) still fails fast with no retry.

Dedup gates and the daily cap stayed in the bot — none of that is
yt-dlp-specific.
"""

import os
import tempfile
import time

import yt_dlp

MAX_FILESIZE_BYTES = 50 * 1024 * 1024  # Telegram Bot API upload cap
SOCKET_TIMEOUT_SECONDS = 20

# Instagram's own access-check API ("get_ruling_for_content") withholds the
# CSRF token an anonymous request needs inconsistently, not per-post — the
# same request retried moments later can succeed. Bounded retry only for
# this exact signature; every other yt-dlp failure still fails fast.
INSTAGRAM_ACCESS_GATE_SIGNAL = "Instagram sent an empty media response"
INSTAGRAM_ACCESS_RETRY_ATTEMPTS = 3
INSTAGRAM_ACCESS_RETRY_BACKOFF_SECONDS = 3


def build_ydl_opts(target_dir: str) -> dict:
    """Assemble yt-dlp options for an anonymous, filesize-capped download.

    Args:
        target_dir: Directory the downloaded file is written into.

    Returns:
        Options dict for ``yt_dlp.YoutubeDL``.
    """
    return {
        "format": f"b[filesize<{MAX_FILESIZE_BYTES}]/b",
        "outtmpl": os.path.join(target_dir, "reel.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": SOCKET_TIMEOUT_SECONDS,
        "max_filesize": MAX_FILESIZE_BYTES,
    }


def extract_info_retrying_access_gate(ydl: yt_dlp.YoutubeDL, url: str) -> dict:
    """Run ``extract_info``, retrying only Instagram's access-gate error.

    Args:
        ydl: Open ``YoutubeDL`` instance to extract with.
        url: Canonical Reel URL.

    Returns:
        yt-dlp's info dict.

    Raises:
        yt_dlp.utils.DownloadError: The access gate persisted through all
            retries, or the failure was some other error (private,
            deleted, unsupported) that is never retried.
    """
    last_error = None
    for attempt in range(INSTAGRAM_ACCESS_RETRY_ATTEMPTS):
        try:
            return ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as err:
            if INSTAGRAM_ACCESS_GATE_SIGNAL not in str(err):
                raise
            last_error = err
            if attempt < INSTAGRAM_ACCESS_RETRY_ATTEMPTS - 1:
                time.sleep(INSTAGRAM_ACCESS_RETRY_BACKOFF_SECONDS)
    raise last_error


def download_reel(url: str) -> tuple[bytes, dict]:
    """Download one Reel into a temporary directory (blocking).

    Args:
        url: Canonical Reel URL.

    Returns:
        Tuple of the downloaded video bytes and yt-dlp's info dict.

    Raises:
        yt_dlp.utils.DownloadError: On extraction or download failure.
        FileNotFoundError: When the filesize guard rejected the video or
            nothing was downloaded.
    """
    with tempfile.TemporaryDirectory() as target_dir:
        with yt_dlp.YoutubeDL(build_ydl_opts(target_dir)) as ydl:
            info = extract_info_retrying_access_gate(ydl, url)
        requested = (info or {}).get("requested_downloads") or []
        filepath = requested[0].get("filepath") if requested else None
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError(f"Reel rejected by filesize guard or not downloaded: {url}")
        with open(filepath, "rb") as video_file:
            return video_file.read(), info
