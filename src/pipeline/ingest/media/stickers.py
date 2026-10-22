"""Sticker descriptions, cached per sticker identity (``file_unique_id``).

A resent sticker never costs a second vision call. Static WEBP stickers are
described directly, video WEBM stickers through keyframe extraction, and
animated Lottie ``.tgs`` stickers keep their placeholder (not renderable)."""

import io

from src import log
from src.pipeline.ingest.media.video import extract_and_describe_frames
from src.pipeline.ingest.media.vision import describe_image_bytes
from src.store import sticker_descriptions

logger = log.get_logger(__name__)


def sticker_kind_from_bytes(raw_bytes: bytes) -> str:
    """Classify a downloaded sticker payload by its magic bytes.

    Args:
        raw_bytes: The sticker file payload.

    Returns:
        ``"video"`` for WEBM (EBML header), ``"animated"`` for gzipped
        Lottie ``.tgs``, otherwise ``"static"`` (WEBP/PNG — describable
        directly as an image).
    """
    if raw_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return "video"
    if raw_bytes[:2] == b"\x1f\x8b":
        return "animated"
    return "static"


async def describe_sticker_bytes(raw_bytes: bytes) -> str:
    """Describe a downloaded sticker payload via the vision LLM.

    Static stickers go straight to the image path; video stickers reuse the
    PyAV keyframe extraction (sticker durations are short, so this yields a
    single middle frame); animated Lottie stickers are not renderable and
    yield an empty description.

    Args:
        raw_bytes: The sticker file payload.

    Returns:
        The vision description, or empty string for animated stickers or
        when frame extraction produced nothing.
    """
    kind = sticker_kind_from_bytes(raw_bytes)
    if kind == "animated":
        return ""
    if kind == "video":
        frame_results = await extract_and_describe_frames(raw_bytes)
        return "\n".join(description for _, description in frame_results)
    _, description = await describe_image_bytes(raw_bytes)
    return description


async def describe_sticker(file_id: str, bot) -> str:
    """Return a vision description for a sticker, cached by its stable identity.

    Resolves the sticker via ``get_file`` to obtain ``file_unique_id`` (stable
    across resends and bots, unlike ``file_id``) and checks the persistent
    cache first — a hit costs no download and no LLM call. On a miss the
    sticker is downloaded and described; a non-empty result is cached so each
    distinct sticker is described at most once ever.

    Args:
        file_id: Telegram file id of the sticker.
        bot: Telegram bot instance used to resolve and download the file.

    Returns:
        The description, or empty string for animated ``.tgs`` stickers and
        on any failure (mirrors ``describe_photo``'s fail-soft contract).
    """
    try:
        tg_file = await bot.get_file(file_id)
        cached = await sticker_descriptions.get_description(tg_file.file_unique_id)
        if cached is not None:
            return cached
        if (tg_file.file_path or "").endswith(".tgs"):
            return ""
        buffer = io.BytesIO()
        await tg_file.download_to_memory(buffer)
        description = await describe_sticker_bytes(buffer.getvalue())
        if description:
            await sticker_descriptions.save_description(tg_file.file_unique_id, description)
        return description
    except Exception as err:
        logger.error("Sticker description failed for file %s: %s", file_id, err)
        return ""
