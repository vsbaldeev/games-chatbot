"""Shared handler abstraction for lightweight social-link summaries.

Instagram Reel and long-form YouTube video links both follow the same
shape: regex-detect a link in chat, gate it (repost dedup + daily cap, same
TtlGate pattern as src.pipeline.shorts), fetch metadata + top comments (no
transcript/audio/vision analysis — that stays exclusive to shorts.py, kept
deliberately separate; see the design doc), and inject a labelled text
block into the pipeline. Instagram alone may also attach downloaded video
bytes for Telegram to repost.

The router consults HANDLERS in priority order: the first handler whose
regex matches anything in the message wins the whole message — every other
link, same platform or different, is ignored (design doc, "Multi-link
messages").
"""

import re
from typing import Protocol, TypedDict

from src.utils.ttl_gate import TtlGate

SOCIAL_LINK_MAX_COMMENTS = 10                  # top-level comments surfaced per handler
SOCIAL_LINK_COMMENT_CHAR_LIMIT = 200           # truncate each comment before prompting
SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT = 2000      # caps YouTube description length
SOCIAL_LINK_DEDUP_WINDOW_SECONDS = 24 * 3600   # same item in the same chat -> one summary
SOCIAL_LINK_DAILY_CAP = 15                     # summaries per chat per sliding 24h window


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
    daily_cap: int

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


def render_comment_lines(comments: list[dict], max_comments: int, char_limit: int) -> str:
    """Render top comments (text + like_count) as a labelled block.

    Shared by instagram_reel.py and youtube_video.py, whose yt-dlp comment
    dicts have the same ``text``/``like_count`` shape.

    Args:
        comments: Comment dicts with ``text`` and ``like_count`` keys.
        max_comments: Maximum number of comments to include.
        char_limit: Truncate each comment's text beyond this length.

    Returns:
        A ``[Топ-комментарии]`` block, or "" when there is nothing usable.
    """
    lines = []
    for comment in comments[:max_comments]:
        text = (comment.get("text") or "").strip()
        if not text:
            continue
        if len(text) > char_limit:
            text = text[:char_limit] + "…"
        like_count = comment.get("like_count") or 0
        lines.append(f"- ({like_count} лайков) {text}")
    if not lines:
        return ""
    return "\n".join(["[Топ-комментарии]:", *lines])


from src.pipeline.social_links import instagram_reel, youtube_video  # noqa: E402

HANDLERS: list[LinkHandler] = [
    instagram_reel.HANDLER,
    youtube_video.HANDLER,
]
