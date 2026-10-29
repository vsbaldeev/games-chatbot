"""Summaries of Shorts and social links (Instagram, TikTok, long-form YouTube, …).

The block is built from the downloaded media's transcript and keyframes plus
the top comments as audience reaction."""

import asyncio

from src import log
from src.pipeline.ingest.links import shorts, social_links
from src.pipeline.ingest.links.comment_summary import summarize_comments
from src.pipeline.ingest.media.transcription import transcribe_bytes
from src.pipeline.ingest.media.video import compose_video_content, extract_and_describe_frames

logger = log.get_logger(__name__)


def compose_short_header(info: dict) -> str:
    """Build the metadata header line for a summarized Short.

    Args:
        info: yt-dlp info dict of the downloaded video.

    Returns:
        A ``[YouTube Shorts …]`` line with whichever of title, channel and
        duration are available.
    """
    details = []
    title = (info.get("title") or "").strip()
    if title:
        details.append(f"«{title}»")
    channel = (info.get("channel") or info.get("uploader") or "").strip()
    if channel:
        details.append(f"канал {channel}")
    duration = info.get("duration")
    if duration:
        details.append(f"{int(duration)} сек")
    suffix = f" {', '.join(details)}" if details else ""
    return f"[YouTube Shorts{suffix}]"


async def summarize_youtube_short(url: str) -> tuple[str, bytes | None]:
    """Download a YouTube Short and return its labelled content block and video bytes.

    Reuses the Telegram-video machinery: Whisper on the audio track and
    vision descriptions of extracted keyframes, plus the top comments as
    audience reaction. The downloaded bytes are returned alongside the text
    so the caller can also repost the video to chat, without a second
    download.

    Args:
        url: Canonical Shorts URL detected by the router.

    Returns:
        ``(content_block, video_bytes)`` on success. ``("", None)`` when the
        download failed or produced neither transcript nor frames — a title
        and comments alone are not enough to react to the video honestly.
    """
    downloaded = await shorts.download_short(url)
    if downloaded is None:
        return "", None
    video_bytes, info = downloaded
    transcribe_result, frame_results = await asyncio.gather(
        transcribe_bytes(video_bytes, "video", "short.mp4"),
        extract_and_describe_frames(video_bytes),
    )
    # unused here — video framing doesn't consume the confidence signal
    # (voice-only, see build_voice_trigger_line)
    transcript, transcript_low_confidence = transcribe_result
    frame_descriptions = [description for _, description in frame_results]
    if not transcript and not frame_descriptions:
        logger.warning("Shorts content extraction produced nothing for %s", url)
        return "", None
    if len(transcript) > shorts.TRANSCRIPT_CHAR_LIMIT:
        transcript = transcript[:shorts.TRANSCRIPT_CHAR_LIMIT] + "…"
    parts = [
        compose_short_header(info or {}),
        compose_video_content(transcript, frame_descriptions),
    ]
    comments_block = await summarize_comments((info or {}).get("comments"))
    if comments_block:
        parts.append(comments_block)
    return "\n".join(parts), video_bytes


async def summarize_social_link(handler_name: str, url: str) -> tuple[str, bytes | None]:
    """Fetch and compose a social-link content block via its matched handler.

    Args:
        handler_name: Name of the handler the router matched
            (``"instagram_reel"``, ``"tiktok_video"`` or
            ``"youtube_video"``).
        url: Canonical URL to fetch, set by the router.

    Returns:
        Tuple of ``(content_block, video_bytes)`` — ``("", None)`` when the
        handler name is unrecognized, the fetch produced nothing usable, or
        the fetch raised (never propagates — same degrade-to-silence
        contract as Shorts).
    """
    handler = next(
        (candidate for candidate in social_links.HANDLERS if candidate.name == handler_name),
        None,
    )
    if handler is None:
        logger.warning("Unknown social link handler %r for %s", handler_name, url)
        return "", None
    try:
        content = await handler.fetch(url)
    except Exception as err:
        logger.warning("Social link handler %r raised for %s: %s", handler_name, url, err)
        return "", None
    if content is None:
        return "", None
    return content["content_block"], content["video_bytes"]
