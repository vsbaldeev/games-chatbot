"""Instagram Reel link detection and anonymous yt-dlp fetch.

Best-effort, no authentication: Instagram aggressively blocks anonymous
scraping, so any failure (blocked, private, deleted, timeout) degrades to
None with no retry — the message falls through to normal routing, same
philosophy as a gated YouTube Shorts repost. Comments are rarely available
via yt-dlp without an authenticated session; when present they are
surfaced, when absent the content block is caption-only.
"""

import asyncio
import os
import re
import tempfile

import yt_dlp

from src import log
from src.pipeline.social_links import (
    SOCIAL_LINK_COMMENT_CHAR_LIMIT,
    SOCIAL_LINK_DAILY_CAP,
    SOCIAL_LINK_DEDUP_WINDOW_SECONDS,
    SOCIAL_LINK_MAX_COMMENTS,
    SocialLinkContent,
    render_comment_lines,
)
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

INSTAGRAM_URL_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/reels?/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)

DOWNLOAD_TIMEOUT_SECONDS = 60
SOCKET_TIMEOUT_SECONDS = 20
MAX_FILESIZE_BYTES = 50 * 1024 * 1024  # Telegram Bot API upload cap


class InstagramReelHandler:
    """Downloads an Instagram Reel anonymously; caption + comments, best effort."""

    name = "instagram_reel"
    pattern = INSTAGRAM_URL_RE

    def __init__(self) -> None:
        """Initialise this handler's own dedup and daily-cap gates."""
        self.dedup_gate = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)
        self.daily_cap_gate = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)
        self.daily_cap = SOCIAL_LINK_DAILY_CAP

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
        """Anonymously download the Reel and compose caption + comments.

        Args:
            url: Canonical Reel URL.

        Returns:
            SocialLinkContent with the downloaded video bytes attached, or
            None on any failure — never retried, never authenticated.
        """
        downloaded = await self.__download(url)
        if downloaded is None:
            return None
        video_bytes, info = downloaded
        content_block = self.__compose(info or {})
        if not content_block:
            return None
        return {"content_block": content_block, "video_bytes": video_bytes}

    def __build_ydl_opts(self, target_dir: str) -> dict:
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

    def __download_sync(self, url: str, target_dir: str) -> tuple[bytes, dict]:
        """Download one Reel into ``target_dir`` (blocking).

        Args:
            url: Canonical Reel URL.
            target_dir: Directory to download the video into.

        Returns:
            Tuple of the downloaded video bytes and yt-dlp's info dict.

        Raises:
            FileNotFoundError: When the filesize guard rejected the video or
                nothing was downloaded.
        """
        with yt_dlp.YoutubeDL(self.__build_ydl_opts(target_dir)) as ydl:
            info = ydl.extract_info(url, download=True)
        requested = (info or {}).get("requested_downloads") or []
        filepath = requested[0].get("filepath") if requested else None
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError(f"Reel rejected by filesize guard or not downloaded: {url}")
        with open(filepath, "rb") as video_file:
            return video_file.read(), info

    async def __download(self, url: str) -> tuple[bytes, dict] | None:
        """Download without blocking the event loop; None on any failure.

        Args:
            url: Canonical Reel URL.

        Returns:
            ``(video_bytes, info_dict)`` on success, None on any failure
            (download error, filesize rejection, timeout) — logged, never
            raised.
        """
        loop = asyncio.get_event_loop()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as target_dir:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, self.__download_sync, url, target_dir),
                    timeout=DOWNLOAD_TIMEOUT_SECONDS,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Instagram Reel download timed out after %ss: %s", DOWNLOAD_TIMEOUT_SECONDS, url
            )
        except Exception as err:
            logger.warning("Instagram Reel download failed for %s: %s", url, err)
        return None

    def __compose(self, info: dict) -> str:
        """Build the labelled content block from yt-dlp's info dict.

        Args:
            info: yt-dlp info dict of the downloaded Reel.

        Returns:
            Labelled block (header + caption + comments), or "" when there
            is no caption to react to.
        """
        caption = (info.get("description") or "").strip()
        if not caption:
            return ""
        parts = ["[Instagram Reel]", caption]
        comments_block = render_comment_lines(
            info.get("comments") or [], SOCIAL_LINK_MAX_COMMENTS, SOCIAL_LINK_COMMENT_CHAR_LIMIT
        )
        if comments_block:
            parts.append(comments_block)
        return "\n".join(parts)


HANDLER = InstagramReelHandler()
