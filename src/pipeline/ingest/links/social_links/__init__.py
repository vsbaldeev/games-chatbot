"""Shared handler abstraction for lightweight social-link summaries.

Instagram Reel, TikTok and long-form YouTube video links all follow the same
shape: regex-detect a link in chat, gate it (repost dedup, same TtlGate
pattern as src.pipeline.ingest.links.shorts), fetch metadata + top comments (no
transcript/audio/vision analysis — that stays exclusive to shorts.py, kept
deliberately separate; see the design doc), and inject a labelled text
block into the pipeline. Instagram and TikTok also attach downloaded video
bytes for Telegram to repost.

Each handler declares its own ``daily_cap`` (None means uncapped). The two
short-video handlers (Instagram, TikTok) count against one shared
``SHORT_VIDEO_DAILY_CAP_GATE`` — a chat gets 30 short-video summaries a day
across both platforms, not 30 each. The YouTube handler is a single free
metadata request with nothing to throttle.

The router consults HANDLERS in priority order: the first handler whose
regex matches anything in the message wins the whole message — every other
link, same platform or different, is ignored (design doc, "Multi-link
messages").
"""

import re
from typing import Protocol, TypedDict

from src.utils.ttl_gate import TtlGate

SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT = 2000      # caps YouTube description length
SOCIAL_LINK_DEDUP_WINDOW_SECONDS = 24 * 3600   # same item in the same chat -> one summary
SHORT_VIDEO_DAILY_CAP = 30                     # Instagram + TikTok summaries per chat per sliding 24h window

# One gate instance shared by every short-video handler, so Reels and TikToks
# draw down the same per-chat budget.
SHORT_VIDEO_DAILY_CAP_GATE = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)


class SocialLinkContent(TypedDict):
    """Composed content from a single social-link fetch."""

    content_block: str
    video_bytes: bytes | None


class LinkHandler(Protocol):
    """One platform's link-detection + fetch + compose logic.

    Implementers are plain classes — no shared base class, just structural
    typing. Each module exposes a module-level ``HANDLER`` singleton
    instance that the registry in this package's ``HANDLERS`` list
    references directly.
    """

    name: str
    pattern: re.Pattern           # used by is_bare_link_message
    dedup_gate: TtlGate
    daily_cap_gate: TtlGate
    daily_cap: int | None  # None disables the cap check entirely

    def extract(self, text: str) -> tuple[str, str] | None:
        """Return ``(item_id, canonical_url)`` for the first match in ``text``, or None."""
        ...

    async def fetch(self, url: str) -> SocialLinkContent | None:
        """Fetch and compose the content block, or None on any failure."""
        ...


# Characters that may legitimately continue a URL past the point the platform
# pattern stops matching: RFC 3986 unreserved + reserved + percent-encoding.
# This is what swallows "?si=..." tracking parameters and trailing slashes.
# Emoji, Cyrillic and ordinary words fall outside it, so content glued
# straight onto a link survives and keeps the message non-bare.
URL_TAIL_RE = re.compile(r"[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*")


def drop_platform_links(text: str, pattern: re.Pattern) -> str:
    """Remove this platform's links, each with its full URL tail.

    Args:
        text: Raw message text.
        pattern: The compiled link pattern identifying the platform.

    Returns:
        The text with every match of ``pattern`` removed together with the
        URL characters trailing it. Links to anywhere else are left in
        place, so a second, unrelated link still counts as content the user
        would lose.
    """
    pieces = []
    cursor = 0
    for match in pattern.finditer(text):
        tail = URL_TAIL_RE.match(text, match.end())
        pieces.append(text[cursor:match.start()])
        cursor = tail.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def is_bare_link_message(text: str | None, pattern: re.Pattern) -> bool:
    """Return whether the message is nothing but a link of this platform.

    Removing the platform's URL tokens from the raw text must leave only
    whitespace. Working on the raw text (not the canonical URL) is what
    handles ``?si=`` tracking parameters, ``m.`` hosts, trailing slashes
    and ``youtu.be`` short forms.

    Deliberately fail-safe: any leftover — a word, an emoji, a second
    link — counts as the user's own content and makes the message
    non-bare, so the delete path never destroys something someone wrote.

    Args:
        text: Raw message text, or None.
        pattern: The compiled link pattern that matched this message.

    Returns:
        True when the message carries the link and nothing else.
    """
    if not text:
        return False
    return not drop_platform_links(text, pattern).strip()


from src.pipeline.ingest.links.social_links import instagram_reel, tiktok_video, youtube_video  # noqa: E402

HANDLERS: list[LinkHandler] = [
    instagram_reel.HANDLER,
    tiktok_video.HANDLER,
    youtube_video.HANDLER,
]
