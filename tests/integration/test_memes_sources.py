"""Integration tests for the live meme sources.

These hit the real 9gag JSON feed and real public ``t.me/s`` channel pages — no
credentials required, but they need network access and depend on external
content, so they are skipped unless ``RUN_MEME_NET_TESTS=1`` is set.

To enable:  RUN_MEME_NET_TESTS=1 pytest tests/integration/test_memes_sources.py
"""

import os

import httpx
import pytest

from src.memes.fetcher import gather_candidates
from src.memes.sources import ninegag, telegram

SKIP_NET = os.environ.get("RUN_MEME_NET_TESTS") != "1"
SKIP_REASON = "Network meme-source tests disabled (set RUN_MEME_NET_TESTS=1 to enable)"

pytestmark = [pytest.mark.integration, pytest.mark.skipif(SKIP_NET, reason=SKIP_REASON)]


def assert_valid_candidate(candidate) -> None:
    """Assert a candidate has a stable key and an https image URL."""
    assert candidate.key
    assert candidate.image_url.startswith("https://")
    assert isinstance(candidate.caption, str)


class TestNinegagLive:
    async def test_fetch_returns_well_formed_photo_candidates(self):
        """9gag may return zero photos on a video-heavy page, but whatever it
        returns must be well-formed photo candidates."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            candidates = await ninegag.fetch(client)
        assert isinstance(candidates, list)
        for candidate in candidates:
            assert_valid_candidate(candidate)
            assert candidate.key.startswith("9gag:")


class TestTelegramLive:
    async def test_known_channel_yields_photos(self):
        """``ru2ch`` is a high-volume public channel that reliably has photos in
        its latest page."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            html = (await client.get(
                "https://t.me/s/ru2ch",
                headers=telegram.BROWSER_HEADERS,
                timeout=10.0,
            )).text
        candidates = telegram.parse_channel(html)
        assert len(candidates) > 0
        for candidate in candidates:
            assert_valid_candidate(candidate)
            assert candidate.key.startswith("tg:ru2ch/")
            assert "telesco.pe" in candidate.image_url or "cdn-telegram.org" in candidate.image_url


class TestGatherCandidatesLive:
    async def test_combines_sources_into_single_list(self):
        candidates = await gather_candidates()
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        for candidate in candidates:
            assert_valid_candidate(candidate)
        keys = [candidate.key for candidate in candidates]
        assert len(set(keys)) == len(keys)
