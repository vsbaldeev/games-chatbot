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

yt-dlp also needs a JS runtime (deno, installed in the Dockerfile) to solve
YouTube's signature/n-parameter challenges; without one it silently falls
back to non-JS player clients. That alone turned out not to be the whole
story: even with deno present, yt-dlp's own default client-selection has
been observed picking a single client ("visionos") whose formats list has
no format 18 at all, failing deterministically rather than flakily.
``SHORTS_PLAYER_CLIENTS`` pins an explicit, known-good client set instead
of trusting that shifting default. Both gaps were only diagnosable via
:class:`YtdlpLogger` below — yt-dlp's own ``quiet``/``no_warnings`` options
were discarding the warnings that actually named each cause.
"""

import asyncio
import os
import re
import tempfile
import time

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

SHORTS_DAILY_CAP = 50               # summaries per chat per sliding 24 h window
DEDUP_WINDOW_SECONDS = 24 * 3600    # same video id in the same chat → one summary

# The bgutil PO-token provider sidecar on the docker-compose network.
POT_PROVIDER_URL = "http://pot-provider:4416"

# Muxed-only selection: format 18 (360p mp4, audio+video in one file) exists
# on virtually every YouTube video; "b" = best pre-muxed fallback. No DASH
# merge → no ffmpeg binary needed in the image.
SHORT_FORMAT = "18/b[ext=mp4][filesize<25M]/b[filesize<25M]"

# yt-dlp auto-selects which of YouTube's several player clients to query,
# and that default has been observed (2026-08-26, via YtdlpLogger below)
# picking a single client — "visionos" — whose formats list has no format
# 18 at all, failing every attempt deterministically rather than flakily.
# yt-dlp's default client set shifts often as it reacts to YouTube's bot
# countermeasures, so instead of trusting whatever it currently prefers,
# pin an explicit set known to carry format 18: android_vr needs no PO
# token at all (REQUIRE_JS_PLAYER=False, no GVS_PO_TOKEN_POLICY); android
# and ios both accept the bgutil-sourced PO token as an alternative to
# sign-in. yt-dlp queries all three and merges their formats, so one
# client lacking 18 (or being blocked) no longer fails the whole request.
SHORTS_PLAYER_CLIENTS = ["android_vr", "android", "ios"]

# Two known intermittent, per-request YouTube extraction flakes that
# self-heal on a re-request moments later — neither is a per-video block:
#   * a signed CDN URL for the chosen format 403s (see yt-dlp issue #17395)
#   * one of SHORTS_PLAYER_CLIENTS times out for this one request, so the
#     merged formats list comes back without format 18 and format
#     selection fails with "Requested format is not available" — the
#     deterministic version of this (yt-dlp defaulting to a single client
#     that never has format 18) is what SHORTS_PLAYER_CLIENTS above fixes;
#     this retry stays as defense-in-depth for a genuine one-off timeout
#     on one of the pinned clients
# Bounded retry only for these exact signals; every other failure (private,
# age-gated, removed, ...) still fails fast with no retry.
SHORTS_CDN_403_SIGNAL = "unable to download video data: HTTP Error 403"
SHORTS_FORMAT_UNAVAILABLE_SIGNAL = "Requested format is not available"
SHORTS_TRANSIENT_RETRY_SIGNALS = (SHORTS_CDN_403_SIGNAL, SHORTS_FORMAT_UNAVAILABLE_SIGNAL)
SHORTS_TRANSIENT_RETRY_ATTEMPTS = 3
SHORTS_TRANSIENT_RETRY_BACKOFF_SECONDS = 3

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


class YtdlpLogger:
    """Routes yt-dlp's internal diagnostics through this module's logger.

    Passing an object here (``ydl_opts["logger"]``) makes yt-dlp deliver
    every warning/error to it instead of stdout/stderr — and, critically,
    ``YoutubeDL.report_warning``/``to_stderr`` check for a logger *before*
    checking ``quiet``/``no_warnings``, so this bypasses that suppression
    entirely. Without it, a PO-token failure or a player-client error (the
    actual reason format 18 goes missing from the merged list) is silently
    discarded, leaving only the final, contextless "Requested format is not
    available" — quiet/no_warnings are still set for stdout hygiene, but
    diagnosis needs what they were swallowing.
    """

    def debug(self, message: str) -> None:
        logger.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        logger.warning("yt-dlp: %s", message)

    def error(self, message: str) -> None:
        logger.error("yt-dlp: %s", message)


def build_ydl_opts(target_dir: str) -> dict:
    """Assemble yt-dlp options for downloading one Short into a directory.

    Args:
        target_dir: Directory the muxed mp4 is written into.

    Returns:
        Options dict for ``yt_dlp.YoutubeDL``: muxed-only format, a pinned
        player-client set (``SHORTS_PLAYER_CLIENTS``), duration and filesize
        guards, top-comments fetching, the PO-token provider address for the
        bgutil plugin, and a logger that surfaces internal yt-dlp warnings
        (PO-token/player-client failures) instead of silently dropping them.
    """
    return {
        "format": SHORT_FORMAT,
        "outtmpl": os.path.join(target_dir, "short.%(ext)s"),
        "logger": YtdlpLogger(),
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
                "player_client": SHORTS_PLAYER_CLIENTS,
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


def extract_info_retrying_transient_errors(ydl: yt_dlp.YoutubeDL, url: str) -> dict:
    """Run ``extract_info``, retrying only YouTube's known transient flakes.

    Args:
        ydl: Open ``YoutubeDL`` instance to extract with.
        url: Canonical Shorts URL.

    Returns:
        yt-dlp's info dict.

    Raises:
        yt_dlp.utils.DownloadError: A transient flake persisted through all
            retries, or the failure was some other error (private,
            age-gated, removed, ...) that is never retried.
    """
    last_error = None
    for attempt in range(SHORTS_TRANSIENT_RETRY_ATTEMPTS):
        try:
            return ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as err:
            if not any(signal in str(err) for signal in SHORTS_TRANSIENT_RETRY_SIGNALS):
                raise
            last_error = err
            if attempt < SHORTS_TRANSIENT_RETRY_ATTEMPTS - 1:
                time.sleep(SHORTS_TRANSIENT_RETRY_BACKOFF_SECONDS)
    raise last_error


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
        info = extract_info_retrying_transient_errors(ydl, url)
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
