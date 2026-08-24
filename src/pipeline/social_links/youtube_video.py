"""Long-form YouTube video link detection and metadata-only fetch.

Regular (non-Shorts) YouTube links: youtube.com/watch?v=... and youtu.be/...
short links. Deliberately lighter than shorts.py's flow — no video/audio
download, no transcript — just yt-dlp's info-extraction (download=False)
for the video's description and top comments. No duration cap: cost is one
metadata request regardless of video length.
"""

import asyncio
import re

import yt_dlp

from src import log
from src.pipeline.shorts import POT_PROVIDER_URL
from src.pipeline.social_links import (
    SOCIAL_LINK_COMMENT_CHAR_LIMIT,
    SOCIAL_LINK_DEDUP_WINDOW_SECONDS,
    SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT,
    SOCIAL_LINK_MAX_COMMENTS,
    SocialLinkContent,
    render_comment_lines,
)
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

# Deliberately does not match /shorts/ paths — shorts.py's own regex owns those.
YOUTUBE_VIDEO_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{6,20})"
    r"|https?://youtu\.be/([A-Za-z0-9_-]{6,20})",
    re.IGNORECASE,
)

FETCH_TIMEOUT_SECONDS = 30


class YoutubeVideoHandler:
    """Fetches a long-form YouTube video's description + top comments, no download."""

    name = "youtube_video"
    pattern = YOUTUBE_VIDEO_URL_RE

    def __init__(self) -> None:
        """Initialise this handler's own repost-dedup gate; no daily cap.

        Unlike Instagram, this fetch never hits a flaky anti-bot gate that
        needs throttling — it's a single free metadata request, so there is
        nothing here worth capping per day.
        """
        self.dedup_gate = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)
        self.daily_cap_gate = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)
        self.daily_cap = None

    def extract(self, text: str) -> tuple[str, str] | None:
        """Extract the video id and canonical watch URL from message text.

        Args:
            text: Raw message text, possibly containing a YouTube link.

        Returns:
            ``(video_id, canonical_watch_url)``, or None when no non-Shorts
            YouTube link is present.
        """
        if not text:
            return None
        match = YOUTUBE_VIDEO_URL_RE.search(text)
        if not match:
            return None
        video_id = match.group(1) or match.group(2)
        return video_id, f"https://www.youtube.com/watch?v={video_id}"

    async def fetch(self, url: str) -> SocialLinkContent | None:
        """Fetch the video's description + top comments, no download.

        Args:
            url: Canonical watch URL.

        Returns:
            SocialLinkContent with ``video_bytes`` always None, or None on
            any extraction failure or when both title and description are
            empty.
        """
        info = await self.__extract_info(url)
        if info is None:
            logger.warning("YouTube metadata extraction returned nothing for %s", url)
            return None
        content_block = self.__compose(info)
        if not content_block:
            logger.warning("YouTube metadata had no usable title/description for %s", url)
            return None
        return {"content_block": content_block, "video_bytes": None}

    def __build_ydl_opts(self) -> dict:
        """yt-dlp options for a metadata-only fetch: no download, top comments."""
        return {
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "getcomments": True,
            "extractor_args": {
                "youtube": {
                    "comment_sort": ["top"],
                    "max_comments": [
                        str(SOCIAL_LINK_MAX_COMMENTS), str(SOCIAL_LINK_MAX_COMMENTS), "0",
                    ],
                },
                "youtubepot-bgutilhttp": {"base_url": [POT_PROVIDER_URL]},
            },
        }

    def __extract_info_sync(self, url: str) -> dict:
        """Blocking metadata-only extraction, run off the event loop."""
        with yt_dlp.YoutubeDL(self.__build_ydl_opts()) as ydl:
            return ydl.extract_info(url, download=False) or {}

    async def __extract_info(self, url: str) -> dict | None:
        """Run the extraction without blocking the event loop; None on failure.

        Args:
            url: Canonical watch URL.

        Returns:
            yt-dlp's info dict, or None on timeout or any extraction error.
        """
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self.__extract_info_sync, url),
                timeout=FETCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "YouTube metadata fetch timed out after %ss: %s", FETCH_TIMEOUT_SECONDS, url
            )
        except Exception as err:
            logger.warning("YouTube metadata fetch failed for %s: %s", url, err)
        return None

    def __compose(self, info: dict) -> str:
        """Build the labelled content block from yt-dlp's info dict.

        Args:
            info: yt-dlp info dict (metadata only, no download).

        Returns:
            Labelled block (header + description, truncated to
            ``SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT``, + comments), or "" when
            there is neither a title nor a description.
        """
        title = (info.get("title") or "").strip()
        description = (info.get("description") or "").strip()
        if not title and not description:
            return ""
        header = f"[YouTube «{title}»]" if title else "[YouTube]"
        parts = [header]
        if description:
            if len(description) > SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT:
                description = description[:SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT] + "…"
            parts.append(description)
        comments_block = render_comment_lines(
            info.get("comments") or [], SOCIAL_LINK_MAX_COMMENTS, SOCIAL_LINK_COMMENT_CHAR_LIMIT
        )
        if comments_block:
            parts.append(comments_block)
        return "\n".join(parts)


HANDLER = YoutubeVideoHandler()
