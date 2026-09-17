"""instagram_reel.py tests — anonymous yt-dlp download + the access-gate retry."""

from unittest.mock import Mock, patch

import pytest
import yt_dlp

from instagram_reel import INSTAGRAM_ACCESS_RETRY_ATTEMPTS, build_ydl_opts, extract_info_retrying_access_gate

SLEEP_PATCH_TARGET = "instagram_reel.time.sleep"

ACCESS_GATE_ERROR = yt_dlp.utils.DownloadError(
    "ERROR: [Instagram] Db58et3OXJx: Instagram sent an empty media response. Check if "
    "this post is accessible in your browser without being logged-in."
)


class TestBuildYdlOpts:
    def test_opts_target_the_given_directory(self, tmp_path):
        opts = build_ydl_opts(str(tmp_path))
        assert opts["outtmpl"] == str(tmp_path / "reel.%(ext)s")
        assert opts["max_filesize"] == 50 * 1024 * 1024


class TestExtractInfoRetryingAccessGate:
    def test_retries_access_gate_error_then_succeeds(self):
        ydl = Mock()
        ydl.extract_info.side_effect = [ACCESS_GATE_ERROR, ACCESS_GATE_ERROR, {"id": "abc"}]
        with patch(SLEEP_PATCH_TARGET):
            info = extract_info_retrying_access_gate(ydl, "https://example.com/reel")
        assert info == {"id": "abc"}
        assert ydl.extract_info.call_count == 3

    def test_gives_up_after_max_attempts(self):
        ydl = Mock()
        ydl.extract_info.side_effect = ACCESS_GATE_ERROR
        with patch(SLEEP_PATCH_TARGET):
            with pytest.raises(yt_dlp.utils.DownloadError):
                extract_info_retrying_access_gate(ydl, "https://example.com/reel")
        assert ydl.extract_info.call_count == INSTAGRAM_ACCESS_RETRY_ATTEMPTS

    def test_does_not_retry_unrelated_download_errors(self):
        ydl = Mock()
        private_post_error = yt_dlp.utils.DownloadError(
            "ERROR: [Instagram] Db58et3OXJx: This post is private."
        )
        ydl.extract_info.side_effect = private_post_error
        with patch(SLEEP_PATCH_TARGET) as mock_sleep:
            with pytest.raises(yt_dlp.utils.DownloadError):
                extract_info_retrying_access_gate(ydl, "https://example.com/reel")
        assert ydl.extract_info.call_count == 1
        mock_sleep.assert_not_called()
