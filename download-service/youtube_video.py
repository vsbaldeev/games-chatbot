"""Metadata-only yt-dlp fetch for long-form YouTube videos (relocated from
src/pipeline/social_links/youtube_video.py).

Deliberately lighter than shorts.py: no video/audio download, no transcript
— just yt-dlp's info-extraction (download=False) for the video's
description and top comments. No duration cap: cost is one metadata
request regardless of video length. Never hits a flaky anti-bot gate that
needs throttling, so no retry logic here (unlike shorts.py/instagram_reel.py).

Dedup gates stayed in the bot — none of that is yt-dlp-specific.
"""

import yt_dlp

from shorts import POT_PROVIDER_URL

MAX_COMMENTS = 10


def build_ydl_opts() -> dict:
    """yt-dlp options for a metadata-only fetch: no download, top comments.

    Returns:
        Options dict for ``yt_dlp.YoutubeDL``.
    """
    return {
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "getcomments": True,
        "extractor_args": {
            "youtube": {
                "comment_sort": ["top"],
                "max_comments": [str(MAX_COMMENTS), str(MAX_COMMENTS), "0"],
            },
            "youtubepot-bgutilhttp": {"base_url": [POT_PROVIDER_URL]},
        },
    }


def fetch_metadata(url: str) -> tuple[None, dict]:
    """Fetch a video's metadata without downloading it (blocking).

    Args:
        url: Canonical watch URL.

    Returns:
        ``(None, info_dict)`` — video_bytes is always None for this kind.

    Raises:
        yt_dlp.utils.DownloadError: On extraction failure.
    """
    with yt_dlp.YoutubeDL(build_ydl_opts()) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    return None, info
