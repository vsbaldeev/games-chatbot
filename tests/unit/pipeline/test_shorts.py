"""shorts.py tests — the intermittent transient-flake retry around extract_info."""

from unittest.mock import Mock, patch

import pytest
import yt_dlp

from src.pipeline.shorts import (
    SHORTS_TRANSIENT_RETRY_ATTEMPTS,
    extract_info_retrying_transient_errors,
)

SLEEP_PATCH_TARGET = "src.pipeline.shorts.time.sleep"

CDN_403_ERROR = yt_dlp.utils.DownloadError(
    "ERROR: unable to download video data: HTTP Error 403: Forbidden"
)

FORMAT_UNAVAILABLE_ERROR = yt_dlp.utils.DownloadError(
    "ERROR: Requested format is not available. Use --list-formats for a list "
    "of available formats"
)


class TestExtractInfoRetryingTransientErrors:
    @pytest.mark.parametrize("transient_error", [CDN_403_ERROR, FORMAT_UNAVAILABLE_ERROR])
    def test_retries_transient_error_then_succeeds(self, transient_error):
        ydl = Mock()
        ydl.extract_info.side_effect = [transient_error, {"id": "abc"}]
        with patch(SLEEP_PATCH_TARGET):
            info = extract_info_retrying_transient_errors(ydl, "https://example.com/shorts/abc")
        assert info == {"id": "abc"}
        assert ydl.extract_info.call_count == 2

    @pytest.mark.parametrize("transient_error", [CDN_403_ERROR, FORMAT_UNAVAILABLE_ERROR])
    def test_gives_up_after_max_attempts(self, transient_error):
        ydl = Mock()
        ydl.extract_info.side_effect = transient_error
        with patch(SLEEP_PATCH_TARGET):
            with pytest.raises(yt_dlp.utils.DownloadError):
                extract_info_retrying_transient_errors(ydl, "https://example.com/shorts/abc")
        assert ydl.extract_info.call_count == SHORTS_TRANSIENT_RETRY_ATTEMPTS

    def test_does_not_retry_unrelated_download_errors(self):
        ydl = Mock()
        private_video_error = yt_dlp.utils.DownloadError("ERROR: Private video")
        ydl.extract_info.side_effect = private_video_error
        with patch(SLEEP_PATCH_TARGET) as mock_sleep:
            with pytest.raises(yt_dlp.utils.DownloadError):
                extract_info_retrying_transient_errors(ydl, "https://example.com/shorts/abc")
        assert ydl.extract_info.call_count == 1
        mock_sleep.assert_not_called()
