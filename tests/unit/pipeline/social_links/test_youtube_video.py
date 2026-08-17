"""YoutubeVideoHandler tests — long-form link detection and metadata-only fetch."""

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline import shorts
from src.pipeline.social_links import SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT
from src.pipeline.social_links.youtube_video import YoutubeVideoHandler


@pytest.fixture
def handler() -> YoutubeVideoHandler:
    return YoutubeVideoHandler()


class TestDailyCap:
    def test_has_no_daily_cap(self, handler):
        assert handler.daily_cap is None


class TestExtract:
    def test_no_link_returns_none(self, handler):
        assert handler.extract("no links here") is None

    def test_watch_url_extracts_id_and_canonical_url(self, handler):
        item_id, url = handler.extract("check https://www.youtube.com/watch?v=dQw4w9WgXcQ out")
        assert item_id == "dQw4w9WgXcQ"
        assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_youtu_be_short_link_extracts_id(self, handler):
        item_id, url = handler.extract("https://youtu.be/dQw4w9WgXcQ")
        assert item_id == "dQw4w9WgXcQ"
        assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_shorts_url_does_not_match(self, handler):
        assert handler.extract("https://www.youtube.com/shorts/dQw4w9WgXcQ") is None


class TestBuildYdlOpts:
    def test_opts_include_pot_provider_arg(self, handler):
        opts = handler._YoutubeVideoHandler__build_ydl_opts()
        assert opts["extractor_args"]["youtubepot-bgutilhttp"]["base_url"] == [
            shorts.POT_PROVIDER_URL
        ]


EXTRACT_INFO_PATCH_TARGET = (
    "src.pipeline.social_links.youtube_video.YoutubeVideoHandler._YoutubeVideoHandler__extract_info"
)


class TestFetch:
    async def test_successful_fetch_composes_description_and_comments(self, handler):
        info = {
            "title": "Почему лава оранжевая",
            "description": "Двухчасовой разбор про майнкрафт-лаву.",
            "comments": [{"text": "наконец кто-то это объяснил", "like_count": 30}],
        }
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=info)):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result is not None
        assert result["video_bytes"] is None
        assert "Двухчасовой разбор про майнкрафт-лаву." in result["content_block"]
        assert "(30 лайков)" in result["content_block"]

    async def test_description_at_limit_is_untouched(self, handler):
        description = "a" * SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT
        info = {"description": description}
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=info)):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert description in result["content_block"]
        assert "…" not in result["content_block"]

    async def test_description_over_limit_is_truncated(self, handler):
        long_description = "a" * (SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT + 1000)
        info = {"description": long_description}
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=info)):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        expected = "a" * SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT + "…"
        assert expected in result["content_block"]
        assert long_description not in result["content_block"]

    async def test_empty_description_with_title_and_comments_still_composes(self, handler):
        info = {
            "title": "Только заголовок",
            "description": "",
            "comments": [{"text": "живое обсуждение", "like_count": 5}],
        }
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=info)):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result is not None
        assert "Только заголовок" in result["content_block"]
        assert "живое обсуждение" in result["content_block"]

    async def test_no_title_and_no_description_returns_none(self, handler):
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value={})):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result is None

    async def test_extraction_failure_returns_none(self, handler):
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=None)):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result is None
