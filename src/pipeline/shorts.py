"""
YouTube Shorts link detection and download.

The router consults :func:`extract_shorts_url` and the two gates below to
decide whether a text message triggers an automatic Shorts summary; the
ingester then calls :func:`download_short` and runs the existing Whisper +
vision pipeline on the downloaded bytes.

This module deliberately does not import from ``src.pipeline.ingester``
(the ingester imports this module — keeps imports acyclic).

Cost gates (both enforced by the router BEFORE any download, so a gated
link costs zero downloads and zero LLM tokens):
  * ``dedup_gate`` — the same video id in the same chat is summarized at
    most once per 24 h (reposts are silently ignored).
  * ``under_daily_cap`` — at most ``SHORTS_DAILY_CAP`` summaries per chat
    per sliding 24 h window, bounding the Whisper/vision token spend.

YouTube bot-detection («Sign in to confirm you're not a bot») on server IPs
is handled automatically by the bgutil PO-token provider: the
``bgutil-ytdlp-pot-provider`` plugin is auto-discovered by yt-dlp at import
and fetches tokens from the ``pot-provider`` docker-compose sidecar (see
``docker-compose.yml``). If the sidecar is down, yt-dlp proceeds without a
token — degraded, never fatal.
"""

import asyncio
import os
import re
import tempfile

import yt_dlp

from src import log
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

SHORTS_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?youtube\.com/shorts/([A-Za-z0-9_-]{6,20})",
    re.IGNORECASE,
)

MAX_SHORT_DURATION_SECONDS = 180              # Shorts hard cap since 2024
MAX_SHORT_FILESIZE_BYTES = 25 * 1024 * 1024   # Groq Whisper upload limit
DOWNLOAD_TIMEOUT_SECONDS = 90
SOCKET_TIMEOUT_SECONDS = 30

MAX_COMMENTS = 10             # top-level comments fetched for audience reaction
COMMENT_CHAR_LIMIT = 200      # truncate each comment before prompting
TRANSCRIPT_CHAR_LIMIT = 2000  # cap speech-dense 3-min shorts before prompting

SHORTS_DAILY_CAP = 15               # summaries per chat per sliding 24 h window
DEDUP_WINDOW_SECONDS = 24 * 3600    # same video id in the same chat → one summary

# The bgutil PO-token provider sidecar on the docker-compose network.
POT_PROVIDER_URL = "http://pot-provider:4416"

# Muxed-only selection: format 18 (360p mp4, audio+video in one file) exists
# on virtually every YouTube video; "b" = best pre-muxed fallback. No DASH
# merge → no ffmpeg binary needed in the image.
SHORT_FORMAT = "18/b[ext=mp4][filesize<25M]/b[filesize<25M]"

# Repost gate: (chat_id, video_id) recorded on first trigger, reposts within
# the window fall through to the normal routing decision.
dedup_gate = TtlGate(DEDUP_WINDOW_SECONDS)

# Token-budget gate: counts summaries per chat in a sliding 24 h window.
daily_cap_gate = TtlGate(DEDUP_WINDOW_SECONDS)


def extract_video_id(text: str | None) -> str | None:
    """Extract the YouTube Shorts video id from a message text.

    Args:
        text: Raw message text, possibly containing a Shorts URL.

    Returns:
        The video id when a Shorts link is present, otherwise ``None``.
    """
    if not text:
        return None
    match = SHORTS_URL_RE.search(text)
    return match.group(1) if match else None


def extract_shorts_url(text: str | None) -> str | None:
    """Extract and canonicalise the first YouTube Shorts URL in a text.

    Canonicalising to ``https://www.youtube.com/shorts/<id>`` strips
    ``?si=`` tracking parameters and normalises the ``m.`` mobile host.

    Args:
        text: Raw message text, possibly containing a Shorts URL.

    Returns:
        The canonical Shorts URL, or ``None`` when no link is present.
    """
    video_id = extract_video_id(text)
    if video_id is None:
        return None
    return f"https://www.youtube.com/shorts/{video_id}"


def under_daily_cap(chat_id: int) -> bool:
    """Record one summary attempt for the chat and check the daily budget.

    Args:
        chat_id: Telegram chat id the Shorts link was posted in.

    Returns:
        ``True`` while the chat is within ``SHORTS_DAILY_CAP`` summaries in
        the sliding 24 h window; ``False`` once the cap is exhausted (the
        link then falls through to normal routing, costing zero tokens).
    """
    used = daily_cap_gate.hit(chat_id)
    if used > SHORTS_DAILY_CAP:
        logger.warning(
            "Shorts daily cap reached for chat %s (%d/%d) — skipping summary",
            chat_id, used, SHORTS_DAILY_CAP,
        )
        return False
    return True


def build_ydl_opts(target_dir: str) -> dict:
    """Assemble yt-dlp options for downloading one Short into a directory.

    Args:
        target_dir: Directory the muxed mp4 is written into.

    Returns:
        Options dict for ``yt_dlp.YoutubeDL``: muxed-only format, duration
        and filesize guards, top-comments fetching and the PO-token provider
        address for the bgutil plugin.
    """
    return {
        "format": SHORT_FORMAT,
        "outtmpl": os.path.join(target_dir, "short.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": SOCKET_TIMEOUT_SECONDS,
        "max_filesize": MAX_SHORT_FILESIZE_BYTES,
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration <= {MAX_SHORT_DURATION_SECONDS}"
        ),
        "getcomments": True,
        "extractor_args": {
            "youtube": {
                "comment_sort": ["top"],
                # Fields: max-comments, max-parents, max-replies — one list
                # element per field (the Python-API equivalent of the CLI's
                # comma-separated syntax). No replies: reactions live in the
                # top-level comments, and one page keeps the fetch fast.
                "max_comments": [str(MAX_COMMENTS), str(MAX_COMMENTS), "0"],
            },
            "youtubepot-bgutilhttp": {"base_url": [POT_PROVIDER_URL]},
        },
    }


def download_short_sync(url: str, target_dir: str) -> tuple[bytes, dict]:
    """Download one YouTube Short into ``target_dir`` (blocking).

    Args:
        url: Canonical Shorts URL.
        target_dir: Directory to download the muxed mp4 into.

    Returns:
        Tuple of the downloaded video bytes and yt-dlp's info dict (title,
        channel, duration, comments, …).

    Raises:
        yt_dlp.utils.DownloadError: On extraction or download failure.
        FileNotFoundError: When the duration/filesize guards rejected the
            video, so no file was produced.
    """
    with yt_dlp.YoutubeDL(build_ydl_opts(target_dir)) as ydl:
        info = ydl.extract_info(url, download=True)
    requested = (info or {}).get("requested_downloads") or []
    filepath = requested[0].get("filepath") if requested else None
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Short rejected by duration/filesize guard or not downloaded: {url}"
        )
    with open(filepath, "rb") as video_file:
        return video_file.read(), info


async def download_short(url: str) -> tuple[bytes, dict] | None:
    """Download one YouTube Short without blocking the event loop.

    Runs :func:`download_short_sync` in the default executor inside a
    temporary directory, bounded by ``DOWNLOAD_TIMEOUT_SECONDS``.

    Args:
        url: Canonical Shorts URL.

    Returns:
        ``(video_bytes, info_dict)`` on success, ``None`` on any failure
        (download error, duration/filesize rejection, timeout) — logged,
        never raised, so the pipeline degrades to silence.
    """
    loop = asyncio.get_event_loop()
    try:
        # ignore_cleanup_errors: on timeout the executor thread may still be
        # writing into the directory while it is being removed.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as target_dir:
            return await asyncio.wait_for(
                loop.run_in_executor(None, download_short_sync, url, target_dir),
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        logger.warning("Shorts download timed out after %ss: %s", DOWNLOAD_TIMEOUT_SECONDS, url)
    except Exception as err:
        # Persistent failures here usually mean YouTube bot-detection or a
        # stale yt-dlp extractor — both self-heal (PO-token sidecar, daily
        # yt-dlp self-update), but the log makes the failing stage visible.
        logger.warning("Shorts download failed for %s: %s", url, err)
    return None
