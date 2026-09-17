"""yt-dlp download logic for YouTube Shorts (relocated from src/pipeline/shorts.py).

Dedup gates, the daily cap, and Whisper/vision processing stayed in the bot
(src/pipeline/shorts.py) — none of that is yt-dlp-specific. This module owns
only the download mechanics.

YouTube bot-detection («Sign in to confirm you're not a bot») on server IPs
is handled automatically by the bgutil PO-token provider: the
``bgutil-ytdlp-pot-provider`` plugin is auto-discovered by yt-dlp at import
and fetches tokens from the ``pot-provider`` docker-compose sidecar. If the
sidecar is down, yt-dlp proceeds without a token — degraded, never fatal.

yt-dlp also needs a JS runtime (deno, installed in the Dockerfile) to solve
YouTube's signature/n-parameter challenges; without one it silently falls
back to non-JS player clients. Deno alone is necessary but not sufficient:
yt-dlp still needs the actual challenge-solving *script* to run in it,
which it will only fetch itself at runtime from GitHub/npm if
``--remote-components`` is passed (running arbitrary fetched JS on every
extraction — not something to enable). The safe alternative is the
``yt-dlp-ejs`` PyPI package, which bundles that script locally and is
exact-pinned by yt-dlp to match its own version; ``requirements.txt``
pulls it in via yt-dlp's ``default`` extra, and ``entrypoint.sh`` /
``ytdlp_update.py`` install with the same extra so it never drifts out of
sync on an auto-update.

That JS-runtime story alone also turned out not to be the whole one:
even with deno (and yt-dlp-ejs) present, yt-dlp's own default
client-selection has been observed picking a single client ("visionos")
whose formats list has no format 18 at all, failing deterministically
rather than flakily. ``SHORTS_PLAYER_CLIENTS`` pins an explicit client set
instead of trusting that shifting default — see its comment for why the
set itself has already needed revising once, and why it is pinned to the
same client family (WEBPO_CLIENTS) the bgutil provider above actually
supports, rather than clients that merely looked token-free at the time.
All of this was only diagnosable via :class:`YtdlpLogger` below — yt-dlp's
own ``quiet``/``no_warnings`` options were discarding the warnings that
actually named each cause.
"""

import logging
import os
import tempfile
import time

import yt_dlp

logger = logging.getLogger("shorts")

MAX_SHORT_DURATION_SECONDS = 180              # Shorts hard cap since 2024
MAX_SHORT_FILESIZE_BYTES = 25 * 1024 * 1024   # Groq Whisper upload limit
SOCKET_TIMEOUT_SECONDS = 30

MAX_COMMENTS = 10             # top-level comments fetched for audience reaction

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
# countermeasures (yt-dlp itself self-updates daily — see ytdlp_update.py
# — so "shifts" means "can change again tomorrow"), so instead of trusting
# whatever it currently prefers, pin an explicit set.
#
# First attempt (2026-08-26) pinned android_vr/android/ios because they
# looked token-free at the time. That broke again within hours: a yt-dlp
# self-update added a GVS PO-token requirement to android_vr, and YouTube's
# ongoing SABR-only rollout (yt-dlp issue #12482) started stripping URLs
# from android/ios formats entirely. The deeper problem: the bgutil
# PO-token provider this module already wires up (POT_PROVIDER_URL above)
# can only ever serve WEBPO_CLIENTS — WEB/MWEB/TVHTML5 and their variants
# (see yt_dlp.extractor.youtube.pot.utils.WEBPO_CLIENTS) — never android/
# ios/android_vr, no matter how healthy the sidecar is. Picking non-web
# clients meant never actually using the token pipeline this bot already
# runs a docker-compose service for.
#
# Pinned to web/mweb (WEBPO_CLIENTS members, so the existing pot-provider
# sidecar can actually authenticate them), plus tv (currently no GVS
# requirement at all, per INNERTUBE_CLIENTS, so a free extra chance) and
# android_vr kept from the first attempt (still lists format 18's URL —
# whether the missing-GVS-token 403 actually fires may not be universal
# or fully rolled out, so it costs little to leave in the merged set).
SHORTS_PLAYER_CLIENTS = ["web", "mweb", "tv", "android_vr"]

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


def download_short(url: str) -> tuple[bytes, dict]:
    """Download one YouTube Short into a temporary directory (blocking).

    Args:
        url: Canonical Shorts URL.

    Returns:
        Tuple of the downloaded video bytes and yt-dlp's info dict (title,
        channel, duration, comments, …).

    Raises:
        yt_dlp.utils.DownloadError: On extraction or download failure.
        FileNotFoundError: When the duration/filesize guards rejected the
            video, so no file was produced.
    """
    with tempfile.TemporaryDirectory() as target_dir:
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
