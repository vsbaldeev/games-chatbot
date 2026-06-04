"""Fetches meme image URLs from public sources and returns one unseen per chat.

Sources (9gag, public Telegram channels) are read without any API key or login;
see :mod:`src.memes.sources`. Each yields direct image URLs, so the meme is sent
by URL while deduplication is keyed on a stable per-post identifier.
"""

import random

import httpx

from src import log
from src.memes.sources import SOURCES, MemeCandidate
from src.memes.store import get_seen_urls, mark_seen

logger = log.get_logger(__name__)


async def gather_candidates() -> list[MemeCandidate]:
    """Collect meme candidates from every registered source.

    A single HTTP client is shared across sources. Individual sources swallow
    and log their own errors, so a failing source yields an empty list rather
    than aborting the others.

    Returns:
        The combined candidates from all sources (possibly empty).
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        candidates: list[MemeCandidate] = []
        for fetch_source in SOURCES:
            candidates.extend(await fetch_source(client))
    return candidates


async def get_meme(chat_id: int) -> tuple[str, str] | None:
    """Pick a random meme not yet sent to the given chat.

    Args:
        chat_id: Telegram chat the meme is destined for.

    Returns:
        An ``(image_url, caption)`` pair for an unseen meme, or ``None`` when no
        candidates could be fetched or all of them were already sent here.
    """
    try:
        candidates = await gather_candidates()
    except Exception as error:
        logger.error("Meme fetch failed for chat %s: %s", chat_id, error)
        return None

    seen = await get_seen_urls(chat_id)
    unseen = [candidate for candidate in candidates if candidate.key not in seen]
    if not unseen:
        return None

    chosen = random.choice(unseen)
    await mark_seen(chat_id, chosen.key)
    return chosen.image_url, chosen.caption
