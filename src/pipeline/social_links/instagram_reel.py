"""Instagram Reel link detection and download-service dispatch.

Best-effort, no authentication: Instagram aggressively blocks anonymous
scraping, so any failure (blocked, private, deleted, timeout) degrades to
None — the message falls through to normal routing, same philosophy as a
gated YouTube Shorts repost. Comments are rarely available via yt-dlp
without an authenticated session; when present they are surfaced, when
absent the content block is caption-only.

The actual yt-dlp download (curl-cffi impersonation, the access-gate retry)
lives in the ``download-service`` sidecar, not in this process — see
``download-service/instagram_reel.py`` and its README for the mechanics.
"""

import re

from src import downloads, log
from src.pipeline.comment_summary import summarize_comments
from src.pipeline.social_links import SOCIAL_LINK_DEDUP_WINDOW_SECONDS, SocialLinkContent
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

INSTAGRAM_URL_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/reels?/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)

# Own cap, not shared with youtube_video: only Instagram fetches hit the
# download service's flaky anonymous access gate, so only Instagram gets
# throttled.
INSTAGRAM_REEL_DAILY_CAP = 30  # summaries per chat per sliding 24h window


class InstagramReelHandler:
    """Downloads an Instagram Reel via the download service; caption + comments, best effort."""

    name = "instagram_reel"
    pattern = INSTAGRAM_URL_RE

    def __init__(self) -> None:
        """Initialise this handler's own dedup and daily-cap gates."""
        self.dedup_gate = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)
        self.daily_cap_gate = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)
        self.daily_cap = INSTAGRAM_REEL_DAILY_CAP

    def extract(self, text: str) -> tuple[str, str] | None:
        """Extract the Reel id and canonical URL from message text.

        Args:
            text: Raw message text, possibly containing a Reel link.

        Returns:
            ``(reel_id, canonical_url)``, or None when no Reel link is present.
        """
        if not text:
            return None
        match = INSTAGRAM_URL_RE.search(text)
        if not match:
            return None
        reel_id = match.group(1)
        return reel_id, f"https://www.instagram.com/reel/{reel_id}/"

    async def fetch(self, url: str) -> SocialLinkContent | None:
        """Fetch the Reel via the download service and compose caption + comments.

        Args:
            url: Canonical Reel URL.

        Returns:
            SocialLinkContent with the downloaded video bytes attached, or
            None on any failure.
        """
        downloaded = await downloads.download_reel(url)
        if downloaded is None:
            logger.warning("Instagram Reel download returned nothing for %s", url)
            return None
        video_bytes, info = downloaded
        info = info or {}
        comments_block = await summarize_comments(info.get("comments"))
        content_block = self.__compose(info, comments_block)
        if not content_block:
            logger.warning("Instagram Reel had no usable caption/comments for %s", url)
            return None
        return {"content_block": content_block, "video_bytes": video_bytes}

    def __compose(self, info: dict, comments_block: str) -> str:
        """Build the labelled content block from the download service's info dict.

        Args:
            info: Info dict returned by the download service.
            comments_block: Pre-computed comment summary block (possibly "").

        Returns:
            Labelled block (header + caption and/or comments), or "" when
            there is neither a caption nor comments to react to.
        """
        caption = (info.get("description") or "").strip()
        if not caption and not comments_block:
            return ""
        parts = ["[Instagram Reel]"]
        if caption:
            parts.append(caption)
        if comments_block:
            parts.append(comments_block)
        return "\n".join(parts)


HANDLER = InstagramReelHandler()
