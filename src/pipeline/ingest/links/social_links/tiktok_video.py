"""TikTok video link detection and download-service dispatch.

Best-effort, no authentication, same shape as instagram_reel: any failure
(blocked, private, deleted, a photo slideshow, timeout) degrades to None —
the message falls through to normal routing. Comments are rarely available
via an anonymous yt-dlp fetch; when present they are surfaced, when absent
the content block is caption-only.

Short links (``vm.tiktok.com``, ``vt.tiktok.com``, ``tiktok.com/t/``) are
passed to the download service as-is — yt-dlp follows the redirect itself.
Their dedup key is the short code, not the video id, so a short link and
the full link to the same video count as two different items: resolving
the redirect here would cost a network call before every gate check.

Counts against the same per-chat daily cap as Instagram Reels
(``SHORT_VIDEO_DAILY_CAP_GATE``). The actual yt-dlp download lives in the
``download-service`` sidecar — see ``download-service/tiktok_video.py``.
"""

import re

from src import downloads, log
from src.pipeline.ingest.links.comment_summary import summarize_comments
from src.pipeline.ingest.links.social_links import (
    SHORT_VIDEO_DAILY_CAP,
    SHORT_VIDEO_DAILY_CAP_GATE,
    SOCIAL_LINK_DEDUP_WINDOW_SECONDS,
    SocialLinkContent,
)
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

# Photo slideshows (``/@user/photo/<id>``) are deliberately not matched:
# they carry no video stream to download or repost.
TIKTOK_URL_RE = re.compile(
    r"https?://(?:"
    r"(?:www\.|m\.)?tiktok\.com/@(?P<user>[\w.-]+)/video/(?P<video_id>\d+)"
    r"|(?P<short_host>vm|vt)\.tiktok\.com/(?P<short_code>[A-Za-z0-9]+)"
    r"|(?:www\.)?tiktok\.com/t/(?P<t_code>[A-Za-z0-9]+)"
    r")",
    re.IGNORECASE,
)


class TiktokVideoHandler:
    """Downloads a TikTok video via the download service; caption + comments, best effort."""

    name = "tiktok_video"
    pattern = TIKTOK_URL_RE

    def __init__(self) -> None:
        """Initialise this handler's own dedup gate and the shared short-video cap."""
        self.dedup_gate = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)
        self.daily_cap_gate = SHORT_VIDEO_DAILY_CAP_GATE
        self.daily_cap = SHORT_VIDEO_DAILY_CAP

    def extract(self, text: str) -> tuple[str, str] | None:
        """Extract the dedup id and canonical URL from message text.

        Args:
            text: Raw message text, possibly containing a TikTok link.

        Returns:
            ``(item_id, canonical_url)`` — the numeric video id for a full
            link, the short code for a short link — or None when no TikTok
            video link is present.
        """
        if not text:
            return None
        match = TIKTOK_URL_RE.search(text)
        if not match:
            return None
        if match["video_id"]:
            return match["video_id"], f"https://www.tiktok.com/@{match['user']}/video/{match['video_id']}"
        if match["short_code"]:
            short_host = match["short_host"].lower()
            return match["short_code"], f"https://{short_host}.tiktok.com/{match['short_code']}/"
        return match["t_code"], f"https://www.tiktok.com/t/{match['t_code']}/"

    async def fetch(self, url: str) -> SocialLinkContent | None:
        """Fetch the video via the download service and compose caption + comments.

        Args:
            url: Canonical TikTok URL.

        Returns:
            SocialLinkContent with the downloaded video bytes attached, or
            None on any failure.
        """
        downloaded = await downloads.download_tiktok(url)
        if downloaded is None:
            logger.warning("TikTok download returned nothing for %s", url)
            return None
        video_bytes, info = downloaded
        info = info or {}
        comments_block = await summarize_comments(info.get("comments"))
        content_block = self.__compose(info, comments_block)
        if not content_block:
            logger.warning("TikTok video had no usable caption/comments for %s", url)
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
        parts = ["[TikTok]"]
        if caption:
            parts.append(caption)
        if comments_block:
            parts.append(comments_block)
        return "\n".join(parts)


HANDLER = TiktokVideoHandler()
