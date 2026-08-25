"""shorts.py tests — the intermittent transient-flake retry around extract_info."""

import logging
from unittest.mock import Mock, patch

import pytest
import yt_dlp

from src.pipeline.shorts import (
    SHORTS_TRANSIENT_RETRY_ATTEMPTS,
    YtdlpLogger,
    build_ydl_opts,
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


class TestYtdlpLoggerSurfacesInternalDiagnostics:
    """Without a custom logger, yt-dlp's quiet/no_warnings options silently
    discard PO-token/player-client failures — the actual reason a format
    goes missing — leaving only the final, contextless format-selection
    error. YtdlpLogger routes those through this module's own logger instead."""

    def test_warning_is_forwarded_to_the_module_logger(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.pipeline.shorts"):
            YtdlpLogger().warning("Error reaching POST /get_pot (caused by TransportError)")
        assert "Error reaching POST /get_pot" in caplog.text

    def test_debug_is_forwarded_to_the_module_logger(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="src.pipeline.shorts"):
            YtdlpLogger().debug("Generating POT via HTTP server")
        assert "Generating POT via HTTP server" in caplog.text

    def test_build_ydl_opts_wires_the_logger_bypassing_quiet_suppression(self):
        """A YoutubeDL logger is checked before quiet/no_warnings, so setting
        it is what actually makes warnings surface despite those flags."""
        opts = build_ydl_opts("/tmp/whatever")
        assert isinstance(opts["logger"], YtdlpLogger)
        assert opts["quiet"] is True
        assert opts["no_warnings"] is True
