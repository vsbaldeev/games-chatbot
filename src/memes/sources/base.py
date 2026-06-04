"""Shared types for meme sources."""

from typing import Awaitable, Callable, NamedTuple

import httpx


class MemeCandidate(NamedTuple):
    """A single meme candidate produced by a source.

    Attributes:
        key: Stable deduplication identifier, e.g. ``"9gag:aBcDe"`` or
            ``"tg:ru2ch/171337"``. Used to remember what was already sent to a
            chat; deliberately not the CDN image URL, which can rotate.
        image_url: Direct, publicly fetchable image URL sent via Telegram.
        caption: Caption or title text for the meme; may be an empty string.
    """

    key: str
    image_url: str
    caption: str


SourceFetcher = Callable[[httpx.AsyncClient], Awaitable[list[MemeCandidate]]]
"""An async callable that fetches candidates from one meme source."""
