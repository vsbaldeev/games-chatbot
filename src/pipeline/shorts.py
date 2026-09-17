"""
YouTube Shorts link detection and download-service dispatch.

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

The actual yt-dlp download (PO-token wiring, JS challenge solving,
player-client pinning, transient-flake retries) lives in the
``download-service`` sidecar, not in this process — see
``download-service/shorts.py`` and its README for the mechanics, and
``src/downloads/README.md`` for the client contract. This module only
detects links, enforces the cost gates, and hands the URL to the client.
"""

import re

from src import downloads, log
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

SHORTS_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?youtube\.com/shorts/([A-Za-z0-9_-]{6,20})",
    re.IGNORECASE,
)

MAX_COMMENTS = 10             # top-level comments fetched for audience reaction
COMMENT_CHAR_LIMIT = 200      # truncate each comment before prompting
TRANSCRIPT_CHAR_LIMIT = 2000  # cap speech-dense 3-min shorts before prompting

SHORTS_DAILY_CAP = 50               # summaries per chat per sliding 24 h window
DEDUP_WINDOW_SECONDS = 24 * 3600    # same video id in the same chat → one summary

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


async def download_short(url: str) -> tuple[bytes, dict] | None:
    """Download one YouTube Short via the download-service sidecar.

    Args:
        url: Canonical Shorts URL.

    Returns:
        ``(video_bytes, info_dict)`` on success, ``None`` on any failure —
        the download service already degrades every internal error
        (extraction, timeout, duration/filesize rejection) to ``None``.
    """
    return await downloads.download_short(url)
