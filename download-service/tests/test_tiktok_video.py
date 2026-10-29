"""tiktok_video.py tests — anonymous yt-dlp download, H.264 preferred."""

from unittest.mock import MagicMock, patch

import pytest

from tiktok_video import MAX_FILESIZE_BYTES, build_ydl_opts, download_tiktok

YOUTUBE_DL_PATCH_TARGET = "tiktok_video.yt_dlp.YoutubeDL"


class TestBuildYdlOpts:
    def test_opts_target_the_given_directory(self, tmp_path):
        opts = build_ydl_opts(str(tmp_path))
        assert opts["outtmpl"] == str(tmp_path / "tiktok.%(ext)s")
        assert opts["max_filesize"] == MAX_FILESIZE_BYTES

    def test_format_prefers_h264_before_any_codec(self, tmp_path):
        selectors = build_ydl_opts(str(tmp_path))["format"].split("/")
        assert selectors[0].startswith("b[vcodec^=h264]")
        assert selectors[-1] == "b"


def mock_youtube_dl(info: dict) -> MagicMock:
    ydl = MagicMock()
    ydl.__enter__.return_value = ydl
    ydl.extract_info.return_value = info
    return MagicMock(return_value=ydl)


class TestDownloadTiktok:
    def test_returns_downloaded_bytes_and_info(self, tmp_path):
        video_path = tmp_path / "tiktok.mp4"
        video_path.write_bytes(b"video bytes")
        info = {"description": "кот", "requested_downloads": [{"filepath": str(video_path)}]}
        with patch(YOUTUBE_DL_PATCH_TARGET, new=mock_youtube_dl(info)):
            video_bytes, returned_info = download_tiktok("https://www.tiktok.com/@cat/video/1")
        assert video_bytes == b"video bytes"
        assert returned_info is info

    def test_nothing_downloaded_raises(self):
        with patch(YOUTUBE_DL_PATCH_TARGET, new=mock_youtube_dl({"description": "кот"})):
            with pytest.raises(FileNotFoundError):
                download_tiktok("https://www.tiktok.com/@cat/video/1")
