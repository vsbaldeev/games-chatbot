"""TiktokVideoHandler tests — link detection and download-service dispatch."""

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline.ingest.links.social_links import SHORT_VIDEO_DAILY_CAP, SHORT_VIDEO_DAILY_CAP_GATE
from src.pipeline.ingest.links.social_links.tiktok_video import TiktokVideoHandler


@pytest.fixture
def handler() -> TiktokVideoHandler:
    return TiktokVideoHandler()


class TestDailyCap:
    def test_shares_the_short_video_cap_with_instagram(self, handler):
        assert handler.daily_cap == SHORT_VIDEO_DAILY_CAP
        assert handler.daily_cap_gate is SHORT_VIDEO_DAILY_CAP_GATE


class TestExtract:
    def test_no_link_returns_none(self, handler):
        assert handler.extract("no links in this message") is None

    def test_full_link_extracts_video_id_and_drops_tracking_params(self, handler):
        item_id, url = handler.extract(
            "смотри https://www.tiktok.com/@cat.lover_1/video/7234567890123456789?is_from_webapp=1&sender_device=pc"
        )
        assert item_id == "7234567890123456789"
        assert url == "https://www.tiktok.com/@cat.lover_1/video/7234567890123456789"

    def test_mobile_host_matches(self, handler):
        item_id, _ = handler.extract("https://m.tiktok.com/@cat/video/7234567890123456789")
        assert item_id == "7234567890123456789"

    @pytest.mark.parametrize("short_host", ["vm", "vt"])
    def test_short_host_link_is_kept_as_is(self, handler, short_host):
        item_id, url = handler.extract(f"https://{short_host}.tiktok.com/ZMabc123/")
        assert item_id == "ZMabc123"
        assert url == f"https://{short_host}.tiktok.com/ZMabc123/"

    def test_t_path_short_link_is_kept_as_is(self, handler):
        item_id, url = handler.extract("https://www.tiktok.com/t/ZTabc123/")
        assert item_id == "ZTabc123"
        assert url == "https://www.tiktok.com/t/ZTabc123/"

    def test_photo_slideshow_is_not_matched(self, handler):
        assert handler.extract("https://www.tiktok.com/@cat/photo/7234567890123456789") is None


DOWNLOAD_PATCH_TARGET = "src.pipeline.ingest.links.social_links.tiktok_video.downloads.download_tiktok"
SUMMARIZE_PATCH_TARGET = "src.pipeline.ingest.links.social_links.tiktok_video.summarize_comments"
TIKTOK_URL = "https://www.tiktok.com/@cat/video/7234567890123456789"


class TestFetch:
    async def test_successful_download_composes_caption_and_comments(self, handler):
        info = {"description": "кот против пылесоса #fyp", "comments": [{"text": "лол", "like_count": 3}]}
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=(b"video bytes", info))), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="[Реакция комментаторов]:\nсмеются")):
            result = await handler.fetch(TIKTOK_URL)
        assert result["video_bytes"] == b"video bytes"
        assert result["content_block"].startswith("[TikTok]")
        assert "кот против пылесоса" in result["content_block"]
        assert "смеются" in result["content_block"]

    async def test_no_caption_no_comments_returns_none(self, handler):
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=(b"video bytes", {}))), \
             patch(SUMMARIZE_PATCH_TARGET, new=AsyncMock(return_value="")):
            assert await handler.fetch(TIKTOK_URL) is None

    async def test_download_failure_returns_none(self, handler):
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=None)):
            assert await handler.fetch(TIKTOK_URL) is None
