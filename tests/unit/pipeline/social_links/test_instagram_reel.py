"""InstagramReelHandler tests — link detection and anonymous yt-dlp fetch."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import yt_dlp

from src.pipeline.social_links.instagram_reel import (
    INSTAGRAM_ACCESS_RETRY_ATTEMPTS,
    InstagramReelHandler,
)


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


SLEEP_PATCH_TARGET = "src.pipeline.social_links.instagram_reel.time.sleep"

ACCESS_GATE_ERROR = yt_dlp.utils.DownloadError(
    "ERROR: [Instagram] Db58et3OXJx: Instagram sent an empty media response. Check if "
    "this post is accessible in your browser without being logged-in."
)


class TestExtractInfoRetry:
    def test_retries_access_gate_error_then_succeeds(self, handler):
        ydl = Mock()
        ydl.extract_info.side_effect = [ACCESS_GATE_ERROR, ACCESS_GATE_ERROR, {"id": "abc"}]
        with patch(SLEEP_PATCH_TARGET):
            info = handler._InstagramReelHandler__extract_info(ydl, "https://example.com/reel")
        assert info == {"id": "abc"}
        assert ydl.extract_info.call_count == 3

    def test_gives_up_after_max_attempts(self, handler):
        ydl = Mock()
        ydl.extract_info.side_effect = ACCESS_GATE_ERROR
        with patch(SLEEP_PATCH_TARGET):
            with pytest.raises(yt_dlp.utils.DownloadError):
                handler._InstagramReelHandler__extract_info(ydl, "https://example.com/reel")
        assert ydl.extract_info.call_count == INSTAGRAM_ACCESS_RETRY_ATTEMPTS

    def test_does_not_retry_unrelated_download_errors(self, handler):
        ydl = Mock()
        private_post_error = yt_dlp.utils.DownloadError(
            "ERROR: [Instagram] Db58et3OXJx: This post is private."
        )
        ydl.extract_info.side_effect = private_post_error
        with patch(SLEEP_PATCH_TARGET) as mock_sleep:
            with pytest.raises(yt_dlp.utils.DownloadError):
                handler._InstagramReelHandler__extract_info(ydl, "https://example.com/reel")
        assert ydl.extract_info.call_count == 1
        mock_sleep.assert_not_called()
