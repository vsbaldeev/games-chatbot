"""Fetches memes from public sources and returns one vetted, unseen image per chat.

Sources (9gag, public Telegram channels) are read without any API key or login;
see :mod:`src.memes.sources`. Deduplication is keyed on a stable per-post
identifier rather than the CDN URL, which can rotate.

Candidates are downloaded here rather than by the caller because the vision
gate in :mod:`src.memes.judge` needs the bytes anyway — so ``get_meme`` hands
back the image itself, already vetted.
"""

import random

import httpx

from src import config, log
from src.memes.judge import score_meme
from src.memes.sources import SOURCES, MemeCandidate
from src.memes.sources.base import BROWSER_HEADERS
from src.memes.store import get_seen_urls, mark_seen

logger = log.get_logger(__name__)


async def download_image(image_url: str) -> bytes | None:
    """Download raw image bytes for direct upload to Telegram.

    Telegram's own URL fetcher is rejected by some meme CDNs (notably
    ``telesco.pe``, which serves Telegram channel media), so the bot downloads
    the bytes itself with a browser User-Agent and uploads them rather than
    handing Telegram a URL it cannot fetch.

    Args:
        image_url: Direct image URL from a meme source.

    Returns:
        The image bytes, or ``None`` if the download failed.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(image_url, headers=BROWSER_HEADERS)
            response.raise_for_status()
            return response.content
    except Exception as error:
        logger.warning("Failed to download meme image %s: %s", image_url, error)
        return None


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


class JudgeUnavailable(Exception):
    """The vision judge returned no verdict, so nothing may be sent.

    Distinct from a rejection: a rejection is a decision about the candidate,
    this is the absence of one. It aborts the vetting loop rather than
    advancing it — see :func:`get_meme`.
    """


async def vet_candidate(chat_id: int, candidate: MemeCandidate) -> bytes | None:
    """Download one candidate and put it past the vision gate.

    A candidate the judge rules on is marked seen either way: a reject is
    burned permanently so the same non-meme never costs a second download and
    vision call. A failed download is not marked — a CDN hiccup says nothing
    about the image.

    Args:
        chat_id: Telegram chat the meme is destined for.
        candidate: The candidate to download and score.

    Returns:
        The image bytes when it passed, or ``None`` when it should be skipped
        because the download failed or the judge rejected it.

    Raises:
        JudgeUnavailable: The judge produced no verdict.
    """
    image = await download_image(candidate.image_url)
    if image is None:
        return None
    score = await score_meme(image)
    if score is None:
        raise JudgeUnavailable(candidate.key)
    await mark_seen(chat_id, candidate.key)
    if score >= config.MEME_JUDGE_PASS_SCORE:
        return image
    logger.info("Rejected non-meme %s for chat %s (score %s)", candidate.key, chat_id, score)
    return None


async def get_meme(chat_id: int) -> bytes | None:
    """Pick, download and vet a meme not yet sent to the given chat.

    Candidates are gathered once and re-picked from that batch, so a retry
    never re-scrapes the sources.

    A judge outage aborts the whole loop and marks nothing seen: an outage is
    not a verdict, and continuing would burn the remaining attempts on a judge
    already known to be down while permanently consuming good candidates.

    Args:
        chat_id: Telegram chat the meme is destined for.

    Returns:
        The vetted image bytes, or ``None`` when nothing could be fetched,
        everything was already sent here, no candidate passed within
        ``MEME_JUDGE_ATTEMPTS`` tries, or the judge was unavailable.
    """
    try:
        candidates = await gather_candidates()
    except Exception as error:
        logger.error("Meme fetch failed for chat %s: %s", chat_id, error)
        return None

    seen = await get_seen_urls(chat_id)
    pool = [candidate for candidate in candidates if candidate.key not in seen]
    for attempt in range(config.MEME_JUDGE_ATTEMPTS):
        if not pool:
            break
        chosen = pool.pop(random.randrange(len(pool)))
        try:
            image = await vet_candidate(chat_id, chosen)
        except JudgeUnavailable:
            logger.warning("Meme judge unavailable on attempt %s for chat %s — sending nothing",
                           attempt + 1, chat_id)
            return None
        if image is not None:
            return image
    return None
