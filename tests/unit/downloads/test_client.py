"""HTTP client tests for the download service — never-raise contract, per-kind timeouts."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.downloads import client

URL_PATCH_TARGET = "src.downloads.client.config.DOWNLOAD_SERVICE_URL"


def mock_async_client(*, json_body=None, raise_error=None):
    """Build a mock usable as `async with httpx.AsyncClient(...) as client`."""
    response = MagicMock()
    response.json.return_value = json_body or {}
    response.raise_for_status.side_effect = raise_error
    instance = MagicMock()
    instance.post = AsyncMock(return_value=response)
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=instance)


class TestPostDownload:
    async def test_disabled_when_url_empty(self):
        with patch(URL_PATCH_TARGET, ""):
            result = await client.post_download("https://x", "youtube_short", 90)
        assert result is None

    async def test_decodes_video_bytes_and_info(self):
        video_b64 = base64.b64encode(b"video bytes").decode()
        mock_client = mock_async_client(json_body={"video_bytes_base64": video_b64, "info": {"id": "abc"}})
        with patch(URL_PATCH_TARGET, "http://download-service:8000"), \
             patch("src.downloads.client.httpx.AsyncClient", mock_client):
            result = await client.post_download("https://x", "youtube_short", 90)
        assert result == (b"video bytes", {"id": "abc"})

    async def test_metadata_only_kind_has_no_video_bytes(self):
        mock_client = mock_async_client(json_body={"video_bytes_base64": None, "info": {"title": "x"}})
        with patch(URL_PATCH_TARGET, "http://download-service:8000"), \
             patch("src.downloads.client.httpx.AsyncClient", mock_client):
            result = await client.post_download("https://x", "youtube_video", 30)
        assert result == (None, {"title": "x"})

    async def test_http_error_returns_none(self):
        mock_client = mock_async_client(
            raise_error=httpx.HTTPStatusError("boom", request=MagicMock(), response=MagicMock())
        )
        with patch(URL_PATCH_TARGET, "http://download-service:8000"), \
             patch("src.downloads.client.httpx.AsyncClient", mock_client):
            result = await client.post_download("https://x", "youtube_short", 90)
        assert result is None


class TestPerKindWrappers:
    async def test_download_short_uses_youtube_short_kind(self):
        with patch("src.downloads.client.post_download", new=AsyncMock(return_value=(b"v", {}))) as mock_post:
            result = await client.download_short("https://x")
        mock_post.assert_awaited_once_with("https://x", "youtube_short", client.SHORTS_TIMEOUT_SECONDS)
        assert result == (b"v", {})

    async def test_download_reel_uses_instagram_reel_kind(self):
        with patch("src.downloads.client.post_download", new=AsyncMock(return_value=(b"v", {}))) as mock_post:
            await client.download_reel("https://x")
        mock_post.assert_awaited_once_with("https://x", "instagram_reel", client.REEL_TIMEOUT_SECONDS)

    async def test_fetch_youtube_video_returns_info_only(self):
        with patch("src.downloads.client.post_download", new=AsyncMock(return_value=(None, {"title": "x"}))):
            result = await client.fetch_youtube_video("https://x")
        assert result == {"title": "x"}

    async def test_fetch_youtube_video_none_on_failure(self):
        with patch("src.downloads.client.post_download", new=AsyncMock(return_value=None)):
            result = await client.fetch_youtube_video("https://x")
        assert result is None
