"""YoutubeVideoHandler tests — long-form link detection and download-service dispatch."""

from unittest.mock import AsyncMock, patch

import pytest

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


EXTRACT_INFO_PATCH_TARGET = "src.pipeline.social_links.youtube_video.downloads.fetch_youtube_video"
SUMMARIZE_PATCH_TARGET = "src.pipeline.social_links.youtube_video.summarize_comments"


class TestFetch:
    async def test_successful_fetch_composes_description_and_comments(self, handler):
        info = {
            "title": "Почему лава оранжевая",
            "description": "Двухчасовой разбор про майнкрафт-лаву.",
            "comments": [{"text": "наконец кто-то это объяснил", "like_count": 30}],
        }
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=info)), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="[Реакция комментаторов]:\nвсе благодарят")):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result is not None
        assert result["video_bytes"] is None
        assert "Двухчасовой разбор про майнкрафт-лаву." in result["content_block"]
        assert "все благодарят" in result["content_block"]

    async def test_description_at_limit_is_untouched(self, handler):
        description = "a" * SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT
        info = {"description": description}
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=info)), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="")):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert description in result["content_block"]
        assert "…" not in result["content_block"]

    async def test_description_over_limit_is_truncated(self, handler):
        long_description = "a" * (SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT + 1000)
        info = {"description": long_description}
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=info)), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="")):
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
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=info)), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="[Реакция комментаторов]:\nбурно обсуждают")):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result is not None
        assert "Только заголовок" in result["content_block"]
        assert "бурно обсуждают" in result["content_block"]

    async def test_no_title_and_no_description_returns_none(self, handler):
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value={})), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="")):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result is None

    async def test_extraction_failure_returns_none(self, handler):
        with patch(EXTRACT_INFO_PATCH_TARGET, new=AsyncMock(return_value=None)):
            result = await handler.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result is None
