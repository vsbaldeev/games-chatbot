"""Reddit post link detection and metadata fetch.

Regex + gating mirror src.pipeline.shorts's Shorts pattern, but the fetch
step hits Reddit's public JSON API instead of downloading anything — no
video/image content is ever pulled, text only, covering every Reddit post
type (text/image/link/video posts) uniformly through the same JSON shape.
"""

import re

import httpx

from src import log
from src.pipeline.social_links import (
    SOCIAL_LINK_COMMENT_CHAR_LIMIT,
    SOCIAL_LINK_DAILY_CAP,
    SOCIAL_LINK_DEDUP_WINDOW_SECONDS,
    SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT,
    SOCIAL_LINK_MAX_COMMENTS,
    SocialLinkContent,
)
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

REDDIT_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.|old\.|m\.)?reddit\.com/r/[^/\s]+/comments/([a-z0-9]+)[^\s]*"
    r"|(?:https?://)?redd\.it/([a-z0-9]+)",
    re.IGNORECASE,
)

REDDIT_USER_AGENT = "games-chatbot:social-link-summary:v1 (contact: repo owner)"
REQUEST_TIMEOUT_SECONDS = 15
DELETED_AUTHORS = frozenset({"[deleted]", "AutoModerator"})
DELETED_BODIES = frozenset({"[deleted]", "[removed]"})


class RedditPostHandler:
    """Fetches a Reddit post's title/selftext + top comments; no media ever."""

    name = "reddit_post"

    def __init__(self) -> None:
        """Initialise this handler's own dedup and daily-cap gates."""
        self.dedup_gate = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)
        self.daily_cap_gate = TtlGate(SOCIAL_LINK_DEDUP_WINDOW_SECONDS)
        self.daily_cap = SOCIAL_LINK_DAILY_CAP

    def extract(self, text: str) -> tuple[str, str] | None:
        """Extract the Reddit post id and a canonical URL from message text.

        Args:
            text: Raw message text, possibly containing a Reddit link.

        Returns:
            ``(post_id, canonical_url)`` — works for both the full
            ``/r/<sub>/comments/<id>/<slug>/`` form (any subdomain, any
            trailing query string/slug) and ``redd.it/<id>`` short links —
            or None when no Reddit link is present. The canonical URL is
            always ``https://www.reddit.com/comments/<post_id>``, Reddit's
            subreddit-agnostic post path, so appending ``.json`` to it
            always resolves regardless of the original link's shape.
        """
        if not text:
            return None
        match = REDDIT_URL_RE.search(text)
        if not match:
            return None
        post_id = match.group(1) or match.group(2)
        return post_id, f"https://www.reddit.com/comments/{post_id}"

    async def fetch(self, url: str) -> SocialLinkContent | None:
        """Fetch a Reddit post's title/selftext + top comments.

        Args:
            url: The Reddit post URL found in the message.

        Returns:
            SocialLinkContent with ``video_bytes`` always None, or None on
            any fetch failure or an unusable/titleless payload.
        """
        payload = await self.__fetch_json(url)
        if payload is None:
            return None
        content_block = self.__compose(payload)
        if not content_block:
            return None
        return {"content_block": content_block, "video_bytes": None}

    async def __fetch_json(self, url: str) -> list | None:
        """GET ``<url>.json`` and return the parsed payload, or None on failure."""
        json_url = url.rstrip("/") + ".json"
        try:
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
                headers={"User-Agent": REDDIT_USER_AGENT},
            ) as client:
                response = await client.get(json_url)
                response.raise_for_status()
                return response.json()
        except Exception as err:
            logger.warning("Reddit fetch failed for %s: %s", url, err)
            return None

    def __compose(self, payload: list) -> str:
        """Build the labelled content block from Reddit's JSON payload.

        Args:
            payload: Parsed two-element JSON array
                ``[post_listing, comments_listing]``.

        Returns:
            Labelled block (header + selftext + top comments), or "" when
            the payload shape is unexpected or the post has no title.
        """
        try:
            post_data = payload[0]["data"]["children"][0]["data"]
        except (KeyError, IndexError, TypeError):
            return ""
        title = (post_data.get("title") or "").strip()
        if not title:
            return ""
        subreddit = post_data.get("subreddit") or ""
        selftext = (post_data.get("selftext") or "").strip()
        header = f"[Reddit r/{subreddit}] «{title}»" if subreddit else f"[Reddit] «{title}»"
        parts = [header]
        if selftext:
            if len(selftext) > SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT:
                selftext = selftext[:SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT] + "…"
            parts.append(selftext)
        comments_block = self.__compose_comments(payload)
        if comments_block:
            parts.append(comments_block)
        return "\n".join(parts)

    def __compose_comments(self, payload: list) -> str:
        """Render top-level comments sorted by score, filtering deleted/AutoMod ones.

        Args:
            payload: Parsed two-element JSON array
                ``[post_listing, comments_listing]``.

        Returns:
            A ``[Топ-комментарии]`` block, or "" when there is nothing usable.
        """
        try:
            comment_children = payload[1]["data"]["children"]
        except (KeyError, IndexError, TypeError):
            return ""
        candidates = []
        for child in comment_children:
            data = child.get("data", {})
            author = data.get("author") or ""
            body = (data.get("body") or "").strip()
            if not body or author in DELETED_AUTHORS or body in DELETED_BODIES:
                continue
            candidates.append((data.get("score") or 0, body))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return self.__render_comment_lines(candidates)

    def __render_comment_lines(self, candidates: list[tuple[int, str]]) -> str:
        """Truncate and label the top-N score-sorted comment candidates."""
        lines = []
        for score, body in candidates[:SOCIAL_LINK_MAX_COMMENTS]:
            if len(body) > SOCIAL_LINK_COMMENT_CHAR_LIMIT:
                body = body[:SOCIAL_LINK_COMMENT_CHAR_LIMIT] + "…"
            lines.append(f"- ({score} очков) {body}")
        if not lines:
            return ""
        return "\n".join(["[Топ-комментарии]:", *lines])


HANDLER = RedditPostHandler()
