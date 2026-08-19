"""shorts.py tests — the intermittent CDN-403 retry around extract_info."""

from unittest.mock import Mock, patch

import pytest
import yt_dlp

from src.pipeline.shorts import (
    SHORTS_CDN_403_RETRY_ATTEMPTS,
    extract_info_retrying_cdn_403,
)

SLEEP_PATCH_TARGET = "src.pipeline.shorts.time.sleep"

CDN_403_ERROR = yt_dlp.utils.DownloadError(
    "ERROR: unable to download video data: HTTP Error 403: Forbidden"
)


class TestExtractInfoRetryingCdn403:
    def test_retries_cdn_403_then_succeeds(self):
        ydl = Mock()
        ydl.extract_info.side_effect = [CDN_403_ERROR, {"id": "abc"}]
        with patch(SLEEP_PATCH_TARGET):
            info = extract_info_retrying_cdn_403(ydl, "https://example.com/shorts/abc")
        assert info == {"id": "abc"}
        assert ydl.extract_info.call_count == 2

    def test_gives_up_after_max_attempts(self):
        ydl = Mock()
        ydl.extract_info.side_effect = CDN_403_ERROR
        with patch(SLEEP_PATCH_TARGET):
            with pytest.raises(yt_dlp.utils.DownloadError):
                extract_info_retrying_cdn_403(ydl, "https://example.com/shorts/abc")
        assert ydl.extract_info.call_count == SHORTS_CDN_403_RETRY_ATTEMPTS

    def test_does_not_retry_unrelated_download_errors(self):
        ydl = Mock()
        private_video_error = yt_dlp.utils.DownloadError("ERROR: Private video")
        ydl.extract_info.side_effect = private_video_error
        with patch(SLEEP_PATCH_TARGET) as mock_sleep:
            with pytest.raises(yt_dlp.utils.DownloadError):
                extract_info_retrying_cdn_403(ydl, "https://example.com/shorts/abc")
        assert ydl.extract_info.call_count == 1
        mock_sleep.assert_not_called()
