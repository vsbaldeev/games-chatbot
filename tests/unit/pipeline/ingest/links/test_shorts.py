"""shorts.py tests — link detection, cost gates, and the download-service dispatch."""

from unittest.mock import AsyncMock, patch

from src.pipeline.ingest.links import shorts


class TestExtractVideoId:
    def test_no_link_returns_none(self):
        assert shorts.extract_video_id("no links here") is None

    def test_extracts_id_from_shorts_url(self):
        assert shorts.extract_video_id("check https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extracts_id_from_mobile_host(self):
        assert shorts.extract_video_id("https://m.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


class TestExtractShortsUrl:
    def test_no_link_returns_none(self):
        assert shorts.extract_shorts_url("no links here") is None

    def test_canonicalises_tracking_params_and_mobile_host(self):
        url = shorts.extract_shorts_url("https://m.youtube.com/shorts/dQw4w9WgXcQ?si=abc123")
        assert url == "https://www.youtube.com/shorts/dQw4w9WgXcQ"


DOWNLOAD_SHORT_TARGET = "src.pipeline.ingest.links.shorts.downloads.download_short"


class TestDownloadShort:
    async def test_delegates_to_the_download_service_client(self):
        with patch(DOWNLOAD_SHORT_TARGET, new=AsyncMock(return_value=(b"video bytes", {"id": "abc"}))) as mock_download:
            result = await shorts.download_short("https://www.youtube.com/shorts/abc")
        mock_download.assert_awaited_once_with("https://www.youtube.com/shorts/abc")
        assert result == (b"video bytes", {"id": "abc"})

    async def test_none_on_service_failure(self):
        with patch(DOWNLOAD_SHORT_TARGET, new=AsyncMock(return_value=None)):
            result = await shorts.download_short("https://www.youtube.com/shorts/abc")
        assert result is None
