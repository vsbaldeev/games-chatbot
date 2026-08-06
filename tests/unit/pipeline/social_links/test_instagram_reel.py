"""InstagramReelHandler tests — link detection and anonymous yt-dlp fetch."""

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline.social_links.instagram_reel import InstagramReelHandler


@pytest.fixture
def handler() -> InstagramReelHandler:
    return InstagramReelHandler()


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


DOWNLOAD_PATCH_TARGET = (
    "src.pipeline.social_links.instagram_reel.InstagramReelHandler._InstagramReelHandler__download"
)


class TestFetch:
    async def test_successful_download_composes_caption_and_comments(self, handler):
        info = {
            "description": "гвоздь молотком в бетон",
            "comments": [{"text": "дизлайк, техника безопасности", "like_count": 12}],
        }
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=(b"video bytes", info))):
            result = await handler.fetch("https://www.instagram.com/reel/Cx7AbCdEfG/")
        assert result is not None
        assert result["video_bytes"] == b"video bytes"
        assert "гвоздь молотком в бетон" in result["content_block"]
        assert "(12 лайков)" in result["content_block"]

    async def test_successful_download_with_no_comments_is_caption_only(self, handler):
        info = {"description": "шаурма на 200 человек"}
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=(b"video bytes", info))):
            result = await handler.fetch("https://www.instagram.com/reel/Cx7AbCdEfG/")
        assert result is not None
        assert result["video_bytes"] == b"video bytes"
        assert "Топ-комментарии" not in result["content_block"]

    async def test_no_caption_returns_none_even_on_successful_download(self, handler):
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=(b"video bytes", {}))):
            result = await handler.fetch("https://www.instagram.com/reel/Cx7AbCdEfG/")
        assert result is None

    async def test_download_failure_returns_none(self, handler):
        with patch(DOWNLOAD_PATCH_TARGET, new=AsyncMock(return_value=None)):
            result = await handler.fetch("https://www.instagram.com/reel/Cx7AbCdEfG/")
        assert result is None


class TestBuildYdlOpts:
    def test_opts_target_the_given_directory(self, handler, tmp_path):
        opts = handler._InstagramReelHandler__build_ydl_opts(str(tmp_path))
        assert opts["outtmpl"] == str(tmp_path / "reel.%(ext)s")
        assert opts["max_filesize"] == 50 * 1024 * 1024
