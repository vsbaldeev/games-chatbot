"""youtube_video.py tests — metadata-only yt-dlp fetch, no download."""

from youtube_video import build_ydl_opts
import shorts


class TestBuildYdlOpts:
    def test_opts_include_pot_provider_arg(self):
        opts = build_ydl_opts()
        assert opts["extractor_args"]["youtubepot-bgutilhttp"]["base_url"] == [shorts.POT_PROVIDER_URL]

    def test_opts_skip_download(self):
        opts = build_ydl_opts()
        assert opts["skip_download"] is True
