"""Fetches meme image URLs from Reddit and returns one unseen per chat."""

import random

import httpx

from src import log
from src.memes.store import get_seen_urls, mark_seen

logger = log.get_logger(__name__)

REQUEST_HEADERS = {"User-Agent": "games-chatbot/1.0"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")

SUBREDDIT_ENDPOINTS = {
    "expectedrussians": "https://www.reddit.com/r/expectedrussians/.json?f=flair_name%3A%22мем%22&limit=100&t=year",
    "ruAsska":         "https://www.reddit.com/r/ruAsska/.json?f=flair_name%3A%22Мем%22&limit=100&t=year",
    "Pikabu":          "https://www.reddit.com/r/Pikabu/search.json?q=flair%3A%D0%9C%D0%B5%D0%BC&sort=hot&restrict_sr=1&limit=100&t=year",
}


def _extract_posts(posts: list) -> list[tuple[str, str]]:
    result = []
    for post in posts:
        post_data = post["data"]
        if post_data.get("is_video") or post_data.get("is_gallery"):
            continue
        url: str = post_data.get("url", "")
        hint: str = post_data.get("post_hint", "")
        if hint == "image" or any(url.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            title: str = post_data.get("title", "")
            result.append((url, title))
    return result


async def fetch_posts() -> list[tuple[str, str]]:
    async with httpx.AsyncClient(headers=REQUEST_HEADERS, follow_redirects=True) as client:
        all_posts = []
        for subreddit, endpoint in SUBREDDIT_ENDPOINTS.items():
            try:
                response = await client.get(endpoint, timeout=10.0)
                response.raise_for_status()
                posts = response.json()["data"]["children"]
                all_posts.extend(_extract_posts(posts))
            except Exception as error:
                logger.warning("Failed to fetch r/%s: %s", subreddit, error)
    return all_posts


async def get_meme(chat_id: int) -> tuple[str, str] | None:
    try:
        all_posts = await fetch_posts()
    except Exception as error:
        logger.error("Reddit fetch failed for chat %s: %s", chat_id, error)
        return None

    seen = await get_seen_urls(chat_id)
    candidates = [(url, title) for url, title in all_posts if url not in seen]
    if not candidates:
        return None

    chosen_url, chosen_title = random.choice(candidates)
    await mark_seen(chat_id, chosen_url)
    return chosen_url, chosen_title
