"""InstagramReelHandler tests — link detection and download-service dispatch."""

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline.social_links.instagram_reel import INSTAGRAM_REEL_DAILY_CAP, InstagramReelHandler


@pytest.fixture
def handler() -> InstagramReelHandler:
    return InstagramReelHandler()


class TestDailyCap:
    def test_daily_cap_is_thirty(self, handler):
        assert handler.daily_cap == INSTAGRAM_REEL_DAILY_CAP == 30


class TestExtract:
    def test_no_link_returns_none(self, handler):
        assert handler.extract("no links in this message") is None

    def test_reel_link_extracts_id_and_canonical_url(self, handler):
        item_id, url = handler.extract("check this https://www.instagram.com/reel/Cx7AbCdEfG/ out")
        assert item_id == "Cx7AbCdEfG"
        assert url == "https://www.instagram.com/reel/Cx7AbCdEfG/"

    def test_reels_plural_form_also_matches(self, handler):
        item_id, _ = handler.extract("https://www.instagram.com/reels/Cx7AbCdEfG/")
        assert item_id == "Cx7AbCdEfG"


DOWNLOAD_PATCH_TARGET = "src.pipeline.social_links.instagram_reel.downloads.download_reel"
SUMMARIZE_PATCH_TARGET = "src.pipeline.social_links.instagram_reel.summarize_comments"


class TestFetch:
    async def test_successful_download_composes_caption_and_comments(self, handler):
        info = {
            "description": "гвоздь молотком в бетон",
            "comments": [{"text": "дизлайк, техника безопасности", "like_count": 12}],
        }
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=(b"video bytes", info))), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="[Реакция комментаторов]:\nхвалят технику")):
            result = await handler.fetch("https://www.instagram.com/reel/Cx7AbCdEfG/")
        assert result is not None
        assert result["video_bytes"] == b"video bytes"
        assert "гвоздь молотком в бетон" in result["content_block"]
        assert "хвалят технику" in result["content_block"]

    async def test_successful_download_with_no_comments_is_caption_only(self, handler):
        info = {"description": "шаурма на 200 человек"}
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=(b"video bytes", info))), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="")):
            result = await handler.fetch("https://www.instagram.com/reel/Cx7AbCdEfG/")
        assert result is not None
        assert result["video_bytes"] == b"video bytes"
        assert "Реакция комментаторов" not in result["content_block"]

    async def test_no_caption_no_comments_returns_none(self, handler):
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=(b"video bytes", {}))), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="")):
            result = await handler.fetch("https://www.instagram.com/reel/Cx7AbCdEfG/")
        assert result is None

    async def test_no_caption_falls_back_to_comments_only(self, handler):
        info = {"comments": [{"text": "дизлайк, техника безопасности", "like_count": 12}]}
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=(b"video bytes", info))), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="[Реакция комментаторов]:\nхвалят технику")):
            result = await handler.fetch("https://www.instagram.com/reel/Cx7AbCdEfG/")
        assert result is not None
        assert "хвалят технику" in result["content_block"]

    async def test_download_failure_returns_none(self, handler):
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=None)):
            result = await handler.fetch("https://www.instagram.com/reel/Cx7AbCdEfG/")
        assert result is None
