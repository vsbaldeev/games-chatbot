"""Unit tests for the meme fetcher orchestration.

Covers:
  - gather_candidates: fan-out over the SOURCES registry
  - get_meme: per-chat deduplication keyed on the stable candidate key
"""

from unittest.mock import AsyncMock, patch

from src.memes import fetcher
from src.memes.sources.base import MemeCandidate

CHAT_ID = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def candidate(key: str, *, url: str = "", caption: str = "cap") -> MemeCandidate:
    """Build a MemeCandidate, defaulting the image URL from the key."""
    return MemeCandidate(key=key, image_url=url or f"https://img/{key}.jpg", caption=caption)


def source_returning(*candidates: MemeCandidate):
    """Build a fake async source fetcher that ignores its client argument."""
    async def fetch(client):
        return list(candidates)
    return fetch


# ---------------------------------------------------------------------------
# gather_candidates
# ---------------------------------------------------------------------------

class TestGatherCandidates:
    async def test_combines_all_sources(self):
        sources = (
            source_returning(candidate("9gag:a"), candidate("9gag:b")),
            source_returning(candidate("tg:chan/1")),
        )
        with patch("src.memes.fetcher.SOURCES", sources):
            result = await fetcher.gather_candidates()
        assert [item.key for item in result] == ["9gag:a", "9gag:b", "tg:chan/1"]

    async def test_no_sources_returns_empty(self):
        with patch("src.memes.fetcher.SOURCES", ()):
            assert await fetcher.gather_candidates() == []


# ---------------------------------------------------------------------------
# get_meme
# ---------------------------------------------------------------------------

class TestGetMeme:
    async def test_returns_image_url_and_caption_for_unseen(self):
        chosen = candidate("9gag:x", url="https://img/x.jpg", caption="hello")
        with patch("src.memes.fetcher.gather_candidates", AsyncMock(return_value=[chosen])), \
             patch("src.memes.fetcher.get_seen_urls", AsyncMock(return_value=set())), \
             patch("src.memes.fetcher.mark_seen", AsyncMock()) as mark_seen:
            result = await fetcher.get_meme(CHAT_ID)
        assert result == ("https://img/x.jpg", "hello")
        mark_seen.assert_awaited_once_with(CHAT_ID, "9gag:x")

    async def test_filters_seen_keys_and_picks_unseen(self):
        seen = candidate("tg:chan/1")
        fresh = candidate("tg:chan/2")
        with patch("src.memes.fetcher.gather_candidates", AsyncMock(return_value=[seen, fresh])), \
             patch("src.memes.fetcher.get_seen_urls", AsyncMock(return_value={"tg:chan/1"})), \
             patch("src.memes.fetcher.mark_seen", AsyncMock()) as mark_seen:
            image_url, _ = await fetcher.get_meme(CHAT_ID)
        assert image_url == fresh.image_url
        mark_seen.assert_awaited_once_with(CHAT_ID, "tg:chan/2")

    async def test_all_seen_returns_none(self):
        items = [candidate("9gag:a"), candidate("9gag:b")]
        with patch("src.memes.fetcher.gather_candidates", AsyncMock(return_value=items)), \
             patch("src.memes.fetcher.get_seen_urls", AsyncMock(return_value={"9gag:a", "9gag:b"})), \
             patch("src.memes.fetcher.mark_seen", AsyncMock()) as mark_seen:
            result = await fetcher.get_meme(CHAT_ID)
        assert result is None
        mark_seen.assert_not_awaited()

    async def test_no_candidates_returns_none(self):
        with patch("src.memes.fetcher.gather_candidates", AsyncMock(return_value=[])), \
             patch("src.memes.fetcher.get_seen_urls", AsyncMock(return_value=set())):
            assert await fetcher.get_meme(CHAT_ID) is None

    async def test_gather_failure_returns_none(self):
        with patch("src.memes.fetcher.gather_candidates", AsyncMock(side_effect=RuntimeError("boom"))):
            assert await fetcher.get_meme(CHAT_ID) is None
