"""Telegram meme source — scrapes public ``t.me/s`` web previews (no auth)."""

import re

import httpx
from bs4 import BeautifulSoup

from src import log
from src.memes.sources.base import MemeCandidate

logger = log.get_logger(__name__)

TELEGRAM_CHANNELS = ("ru2ch", "memes")
CHANNEL_URL = "https://t.me/s/{channel}"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}
BACKGROUND_IMAGE_PATTERN = re.compile(r"background-image:url\('([^']+)'\)")


def parse_channel(html: str) -> list[MemeCandidate]:
    """Extract standalone-photo memes from a ``t.me/s`` channel page.

    Only messages that carry a single photo are kept; video messages
    (``video_thumb`` posters) and link previews use different markup and are
    skipped, so we never send a video still as if it were a photo.

    Args:
        html: Raw HTML of the channel's public web-preview page.

    Returns:
        One candidate per photo message, keyed by its stable ``data-post``
        permalink (e.g. ``"tg:ru2ch/171337"``).
    """
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[MemeCandidate] = []
    for message in soup.select("div.tgme_widget_message[data-post]"):
        photo = message.select_one("a.tgme_widget_message_photo_wrap")
        if photo is None:
            continue
        match = BACKGROUND_IMAGE_PATTERN.search(photo.get("style", ""))
        if not match:
            continue
        text = message.select_one(".tgme_widget_message_text")
        candidates.append(MemeCandidate(
            key=f"tg:{message['data-post']}",
            image_url=match.group(1),
            caption=text.get_text(strip=True) if text else "",
        ))
    return candidates


async def fetch(client: httpx.AsyncClient) -> list[MemeCandidate]:
    """Fetch photo memes from every configured public Telegram channel.

    Args:
        client: Shared async HTTP client.

    Returns:
        Combined candidates across all channels; a channel that fails is logged
        and skipped so one bad channel never empties the batch.
    """
    candidates: list[MemeCandidate] = []
    for channel in TELEGRAM_CHANNELS:
        try:
            response = await client.get(
                CHANNEL_URL.format(channel=channel),
                headers=BROWSER_HEADERS,
                timeout=10.0,
            )
            response.raise_for_status()
            candidates.extend(parse_channel(response.text))
        except Exception as error:
            logger.warning("Failed to fetch t.me/s/%s: %s", channel, error)
    return candidates
