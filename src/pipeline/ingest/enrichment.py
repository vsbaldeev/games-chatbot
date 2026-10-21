"""Lazy enrichment of stored rows still in placeholder form.

``enrich_media_row`` is the single authoritative entry point (photos via
``enrich_photo_row``, stickers via ``enrich_sticker_row``), used by the filter's
classify node (before classifying a reply to media) and the context builder
(reply chains). Results are cached back to ``unified_messages``."""

from src import log
from src.pipeline.ingest.stickers import describe_sticker
from src.pipeline.ingest.vision import describe_photo
from src.store import unified_messages

logger = log.get_logger(__name__)


async def enrich_photo_row(row: dict, chat_id: int, bot) -> dict:
    """Lazily replace a photo row's placeholder content with a vision description.

    The single authoritative photo-enrichment helper, shared by the filter
    node (before classifying a reply to a photo) and the context builder
    (reply chains). The enriched content is cached back to unified_messages,
    so the vision call happens at most once per photo regardless of caller.

    Args:
        row: A unified_messages-shaped row — stored or synthesized from the
            Telegram update as a fallback.
        chat_id: Chat the row belongs to, used to cache the result.
        bot: Telegram bot instance used to download the photo.

    Returns:
        The row unchanged when it is not a placeholder-form photo, has no
        file_id (fallback rows), or the vision call fails; otherwise a copy
        whose content is the description combined with the original caption.
        A cache-write failure is logged and non-fatal.
    """
    if row["media_type"] != "photo" or not unified_messages.needs_photo_description(row["content"]):
        return row
    file_id = row.get("file_id")
    if not file_id:
        return row
    caption = unified_messages.extract_photo_caption(row["content"])
    _, description = await describe_photo(file_id, bot)
    if not description:
        return row
    combined = unified_messages.combine_description_and_caption(description, caption)
    try:
        await unified_messages.update_content(
            chat_id=chat_id,
            message_id=row["message_id"],
            content=combined,
        )
    except Exception as err:
        logger.warning("Failed to cache photo description for msg %s: %s", row["message_id"], err)
    return {**row, "content": combined}


async def enrich_sticker_row(row: dict, chat_id: int, bot) -> dict:
    """Lazily replace a sticker row's placeholder content with a vision description.

    Mirrors ``enrich_photo_row``: shared by the filter node and the context
    builder via ``enrich_media_row``. The description comes from the
    per-sticker persistent cache when available, and the enriched content is
    written back to unified_messages.

    Args:
        row: A unified_messages-shaped row — stored or synthesized from the
            Telegram update as a fallback.
        chat_id: Chat the row belongs to, used to cache the result.
        bot: Telegram bot instance used to download the sticker.

    Returns:
        The row unchanged when it is not a placeholder-form sticker, has no
        file_id (fallback rows), or no description could be produced;
        otherwise a copy whose content is the description. A cache-write
        failure is logged and non-fatal.
    """
    if row["media_type"] != "sticker" or row["content"] != unified_messages.STICKER_PLACEHOLDER:
        return row
    file_id = row.get("file_id")
    if not file_id:
        return row
    description = await describe_sticker(file_id, bot)
    if not description:
        return row
    try:
        await unified_messages.update_content(
            chat_id=chat_id,
            message_id=row["message_id"],
            content=description,
        )
    except Exception as err:
        logger.warning("Failed to cache sticker description for msg %s: %s", row["message_id"], err)
    return {**row, "content": description}


async def enrich_media_row(row: dict, chat_id: int, bot) -> dict:
    """Lazily vision-enrich a placeholder-form media row, dispatching by media type.

    The single plug-in point for reply-chain and replied-to enrichment: photos
    go through ``enrich_photo_row``, stickers through ``enrich_sticker_row``,
    and every other row passes through untouched.

    Args:
        row: A unified_messages-shaped row.
        chat_id: Chat the row belongs to.
        bot: Telegram bot instance used to download media.

    Returns:
        The (possibly enriched) row.
    """
    if row["media_type"] == "photo":
        return await enrich_photo_row(row, chat_id, bot)
    if row["media_type"] == "sticker":
        return await enrich_sticker_row(row, chat_id, bot)
    return row
