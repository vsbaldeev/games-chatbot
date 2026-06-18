"""Meme source registry.

Each source module exposes an async ``fetch(client)`` returning a list of
:class:`MemeCandidate`. Adding a new source means adding a module and listing
its ``fetch`` here — no changes to the fetcher or handler (Open/Closed).
"""

from src.memes.sources import ninegag, telegram
from src.memes.sources.base import MemeCandidate, SourceFetcher

SOURCES: tuple[SourceFetcher, ...] = (ninegag.fetch, telegram.fetch)

__all__ = ["SOURCES", "MemeCandidate", "SourceFetcher"]
