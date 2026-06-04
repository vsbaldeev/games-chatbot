"""9gag meme source — reads the public group-posts JSON feed (no auth)."""

import httpx

from src import log
from src.memes.sources.base import MemeCandidate

logger = log.get_logger(__name__)

NINEGAG_FEED = "https://9gag.com/v1/group-posts/group/default/type/hot"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}


def parse_ninegag(payload: dict) -> list[MemeCandidate]:
    """Extract still-photo memes from a 9gag group-posts JSON payload.

    Animated posts, videos and NSFW posts are skipped so the result is safe to
    send straight to a chat via ``reply_photo``.

    Args:
        payload: Decoded JSON returned by the 9gag group-posts feed.

    Returns:
        Photo candidates with stable keys and direct ``9cache.com`` image URLs.
    """
    candidates: list[MemeCandidate] = []
    for post in payload.get("data", {}).get("posts", []):
        if post.get("type") != "Photo" or post.get("nsfw"):
            continue
        image_url = post.get("images", {}).get("image700", {}).get("url", "")
        if not image_url:
            continue
        candidates.append(MemeCandidate(
            key=f"9gag:{post.get('id') or image_url}",
            image_url=image_url,
            caption=post.get("title", ""),
        ))
    return candidates


async def fetch(client: httpx.AsyncClient) -> list[MemeCandidate]:
    """Fetch hot photo memes from 9gag.

    Args:
        client: Shared async HTTP client.

    Returns:
        Photo candidates, or an empty list if the feed cannot be read.
    """
    try:
        response = await client.get(NINEGAG_FEED, headers=BROWSER_HEADERS, timeout=10.0)
        response.raise_for_status()
        return parse_ninegag(response.json())
    except Exception as error:
        logger.warning("Failed to fetch 9gag feed: %s", error)
        return []
