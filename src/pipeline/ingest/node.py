"""
MessageIngester — the graph node that turns media and links into readable text.

Each media type has one strategy (see ``INGEST_STRATEGIES``):
  - text       → passed through unchanged, unless the router detected a
                 YouTube Shorts / social link: then the link is downloaded and
                 summarized (see ``links``) and the block is appended to the
                 user's text
  - voice      → transcribed via Groq Whisper
  - video_note → transcribed via Groq Whisper + frames described via Vision LLM
  - video      → same as video_note
  - photo      → described via vision LLM, combined with the caption
  - sticker    → described via vision LLM, but only when the bot is about to
                 respond
  - anything else → the raw text, unchanged

Updates the unified_messages row written by the Router with the real content
so that reply-chain queries later in the pipeline return useful text.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from src import log
from src.pipeline.ingest.links import summarize_social_link, summarize_youtube_short
from src.pipeline.ingest.stickers import describe_sticker
from src.pipeline.ingest.transcription import transcribe_voice
from src.pipeline.ingest.video import transcribe_video
from src.pipeline.ingest.vision import describe_photo
from src.pipeline.state import BotState
from src.store import unified_messages

logger = log.get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """What one media strategy produced for a message."""

    processed: str
    is_real_person: bool | None = None
    extra: dict = field(default_factory=dict)


async def ingest_text(state: BotState) -> IngestResult:
    """Process a text message, summarizing a Shorts or social link when routed so.

    Args:
        state: Current pipeline state.

    Returns:
        The user text with the labelled summary block appended on success;
        ``extra`` carries the fetched content fields and is empty when there
        was no link or the fetch produced nothing usable.
    """
    raw_text = state["incoming"]["raw_text"] or ""
    short_url = state.get("youtube_short_url")
    if short_url:
        return await ingest_shorts(raw_text, short_url)
    social_handler = state.get("social_link_handler")
    social_url = state.get("social_link_url")
    if social_handler and social_url:
        return await ingest_social_link(raw_text, social_handler, social_url)
    return IngestResult(raw_text)


async def ingest_shorts(raw_text: str, short_url: str) -> IngestResult:
    """Summarize a Shorts link and append it to the raw text.

    Args:
        raw_text: The sender's original text (containing the link).
        short_url: Canonical Shorts URL set by the router.

    Returns:
        The combined text with the summary fields; the raw text and no extra
        fields on a failed summary.
    """
    short_content, short_video = await summarize_youtube_short(short_url)
    if not short_content:
        return IngestResult(raw_text)
    combined = f"{raw_text}\n\n{short_content}".strip()
    return IngestResult(
        combined,
        extra={"youtube_short_content": short_content, "youtube_short_video": short_video},
    )


async def ingest_social_link(raw_text: str, handler_name: str, url: str) -> IngestResult:
    """Summarize an Instagram/YouTube link and append it to the raw text.

    Args:
        raw_text: The sender's original text (containing the link).
        handler_name: Matched handler name set by the router.
        url: Canonical URL set by the router.

    Returns:
        The combined text with the summary fields; the raw text and no extra
        fields on a failed fetch.
    """
    social_content, social_video = await summarize_social_link(handler_name, url)
    if not social_content:
        return IngestResult(raw_text)
    combined = f"{raw_text}\n\n{social_content}".strip()
    return IngestResult(
        combined,
        extra={"social_link_content": social_content, "social_link_video": social_video},
    )


async def ingest_voice(state: BotState) -> IngestResult:
    """Transcribe a voice note; records whether Whisper was unsure of it."""
    msg = state["incoming"]
    bot = state["context_types"].bot
    transcript, low_confidence = await transcribe_voice(msg["file_id"], "voice", bot)
    return IngestResult(transcript, extra={"voice_low_confidence": low_confidence})


async def ingest_video(state: BotState) -> IngestResult:
    """Transcribe a video or video note and describe its keyframes."""
    msg = state["incoming"]
    bot = state["context_types"].bot
    is_real_person, content = await transcribe_video(msg["file_id"], msg["media_type"], bot)
    return IngestResult(content, is_real_person)


async def ingest_photo(state: BotState) -> IngestResult:
    """Describe an incoming photo and combine it with its caption.

    Args:
        state: Current pipeline state with a photo message.

    Returns:
        The description combined with the caption; the bare caption when the
        vision call produced no description.
    """
    msg = state["incoming"]
    bot = state["context_types"].bot
    is_real_person, description = await describe_photo(msg["file_id"], bot)
    caption = msg["raw_text"] or ""
    if description:
        combined = unified_messages.combine_description_and_caption(description, caption)
        return IngestResult(combined, is_real_person)
    return IngestResult(caption, is_real_person)


async def ingest_sticker(state: BotState) -> IngestResult:
    """Describe a sticker, but only when the bot is about to answer.

    Plain sticker traffic is enriched lazily (reply chains / replied-to
    lookups) to avoid a vision call per first-seen sticker in the chat.
    """
    msg = state["incoming"]
    if not state["should_respond"]:
        return IngestResult(msg["raw_text"] or "")
    bot = state["context_types"].bot
    return IngestResult(await describe_sticker(msg["file_id"], bot))


async def ingest_raw(state: BotState) -> IngestResult:
    """Fallback for media types with no strategy: keep the raw text."""
    return IngestResult(state["incoming"]["raw_text"] or "")


INGEST_STRATEGIES: dict[str, Callable[[BotState], Awaitable[IngestResult]]] = {
    "text": ingest_text,
    "voice": ingest_voice,
    "video_note": ingest_video,
    "video": ingest_video,
    "photo": ingest_photo,
    "sticker": ingest_sticker,
}


async def update_stored_content(msg: dict, content: str) -> None:
    """Overwrite the stored placeholder row with the real content.

    Args:
        msg: IncomingMessage dict of the message being enriched.
        content: The processed text to store.
    """
    try:
        await unified_messages.update_content(
            chat_id=msg["chat_id"],
            message_id=msg["message_id"],
            content=content,
        )
    except Exception as err:  # noqa: BLE001 — a failed cache write must not fail the run
        logger.warning("Failed to update message content for %s: %s", msg["message_id"], err)


class MessageIngester:
    """Converts media messages to text and updates the unified_messages store."""

    async def __call__(self, state: BotState) -> dict:
        """Convert the incoming message's media (or Shorts/social link) to text.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict with the enriched ``incoming`` message, the
            ``media_is_real_person`` classification (photo/video_note/video
            only; None otherwise) and, for Shorts/social-link triggers, the
            fetched content fields.
        """
        msg = state["incoming"]
        media_type = msg["media_type"]
        result = await INGEST_STRATEGIES.get(media_type, ingest_raw)(state)

        # Media rows are stored as placeholders by the router and only become
        # readable once transcribed/described here. Text rows are already
        # stored verbatim, and a link trigger's processed text additionally
        # carries the fetched material block — which must stay out of chat
        # history, or the next link's prompt shows two identically labelled
        # material blocks and the model retells both. The material is still
        # persisted for reply grounding, in the ``link_material`` column of
        # the bot's own reply row (see ``src.events.messages``).
        if media_type != "text" and result.processed:
            await update_stored_content(msg, result.processed)

        incoming_update = dict(msg)
        incoming_update["processed_text"] = result.processed
        return {
            "incoming": incoming_update,
            "media_is_real_person": result.is_real_person,
            **result.extra,
        }
