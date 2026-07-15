"""
MessageIngester — second node in the LangGraph pipeline.

Processes raw media into readable text:
  - text       → passed through unchanged, unless the router detected a
                 YouTube Shorts link: then the short is downloaded via
                 yt-dlp and summarized with the same Whisper + vision
                 machinery (plus top comments as audience reaction), and
                 the block is appended to the user's text
  - voice      → transcribed via Groq Whisper
  - video_note → transcribed via Groq Whisper + frames described via Vision LLM
  - video      → transcribed via Groq Whisper + frames described via Vision LLM
  - photo      → described via vision LLM (one-sentence description)

  Every vision call also returns a real-person-vs-meme classification (see
  ``parse_vision_response``), piggybacked on the same call — no extra LLM
  round trip. It is surfaced as ``media_is_real_person`` in the pipeline
  state and used by the filter node to skip the unprompted random reaction
  to photos/video notes that are not a genuine photo of a real person
  (memes, screenshots, art…); explicit @mentions/replies are unaffected.
  - sticker    → described via vision LLM, but only when the bot is about to
                 respond; descriptions are cached per sticker identity
                 (file_unique_id) in sticker_descriptions, so a resent
                 sticker never costs a second vision call

Transcription is pinned to Russian (``config.WHISPER_LANGUAGE``) and runs a
garbage-transcript check on Whisper's segment metadata plus a boilerplate
denylist: silence/noise hallucinations («Продолжение следует…», «Спасибо за
просмотр») are rejected and treated as empty transcripts.

Frame extraction (PyAV):
  duration < 15s   → 1 keyframe (middle)
  15s – 120s       → 3 keyframes (uniformly distributed)
  > 120s           → audio only, no frames

Updates the unified_messages row written by the Router with the real content
so that reply-chain queries later in the pipeline return useful text.

Also exports ``enrich_media_row`` — the single authoritative lazy enrichment
for stored rows still in placeholder form (photos via ``enrich_photo_row``,
stickers via ``enrich_sticker_row``), used both by the filter node (before
classifying a reply to media) and the context builder (reply chains).
Results are cached back to unified_messages; sticker descriptions are
additionally cached per file_unique_id, since the same stickers are resent
constantly. Static WEBP stickers are described directly, video WEBM stickers
through keyframe extraction, and animated Lottie ``.tgs`` stickers keep their
placeholder (not renderable).
"""

import asyncio
import base64
import io
import re

from src import log

import av
from groq import AsyncGroq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src import config
from src.agent import ainvoke_with_backoff
from src.config.prompts import VISION_MEME_TAG, VISION_PROMPT, VISION_REAL_PERSON_TAG
from src.pipeline import shorts
from src.pipeline.state import BotState
from src.store import sticker_descriptions, unified_messages

logger = log.get_logger(__name__)

VISION_TAG_RE = re.compile(rf"^\[({VISION_REAL_PERSON_TAG}|{VISION_MEME_TAG})\]\s*", re.IGNORECASE)


def parse_vision_response(raw: str) -> tuple[bool | None, str]:
    """Split a vision LLM response into its real-person classification and description.

    Args:
        raw: Raw vision LLM output, expected to start with a
            ``[ЧЕЛОВЕК]``/``[МЕМ]`` tag per ``VISION_PROMPT``.

    Returns:
        Tuple of ``(is_real_person, description)``. ``is_real_person`` is
        True/False when the tag is present and recognized, or None when the
        model omitted or malformed it — callers gating on this value must
        fail open (None never suppresses a response) rather than treat a
        parsing hiccup as a meme.
    """
    text = raw.strip()
    match = VISION_TAG_RE.match(text)
    if not match:
        return None, text
    is_real_person = match.group(1).upper() == VISION_REAL_PERSON_TAG
    return is_real_person, text[match.end():].strip()


def aggregate_real_person(frame_results: list[tuple[bool | None, str]]) -> bool | None:
    """Aggregate per-frame real-person classifications into one verdict.

    Args:
        frame_results: ``(is_real_person, description)`` pairs, one per
            successfully described keyframe.

    Returns:
        True when a majority of classified frames show a real person, False
        when a majority do not, or None when no frame could be classified —
        callers must treat None as "unknown" and not suppress a response on it.
    """
    votes = [is_real_person for is_real_person, _ in frame_results if is_real_person is not None]
    if not votes:
        return None
    return sum(votes) > len(votes) / 2

def _make_vision_llm() -> ChatGroq:
    """Return a ChatGroq instance configured for vision tasks.

    VISION_MODEL is a reasoning model; without ``reasoning_effort="none"``
    it spends the whole max_tokens budget inside a ``<think>`` block and the
    "description" comes back as truncated reasoning text.
    """
    return ChatGroq(
        model=config.VISION_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=200,
        max_retries=0,
        reasoning_effort="none",
    )


FRAME_DURATION_AUDIO_ONLY = 120
FRAME_DURATION_SINGLE = 15

WHISPER_TIMEOUT = 60.0   # seconds; Groq Whisper usually responds in 2–15 s
WHISPER_RETRIES = 2

# Garbage-transcript thresholds — tune from the rejection info-logs.
NO_SPEECH_PROB_LIMIT = 0.6    # above on every segment → no actual speech
AVG_LOGPROB_LIMIT = -1.0      # below on every segment → low-confidence decode
COMPRESSION_RATIO_LIMIT = 2.4  # above on every segment → repetition loop
BOILERPLATE_MAX_LENGTH = 60   # denylist applies only to short transcripts

# Stock phrases Whisper hallucinates on silence/music with high confidence —
# segment metadata alone does not catch them (verified: 1 s of silence yields
# «Продолжение следует...» at avg_logprob −0.27).
BOILERPLATE_RE = re.compile(
    r"субтитры|спасибо за просмотр|продолжение следует|подписывайтесь на канал"
    r"|thanks for watching|subtitles by",
    re.IGNORECASE,
)


def is_garbage_transcript(text: str, segments: list[dict]) -> bool:
    """Detect Whisper hallucinations produced by silence, noise or music.

    Signals (any one classifies the transcript as garbage):
      1. Every segment has ``no_speech_prob`` above ``NO_SPEECH_PROB_LIMIT``.
      2. Every segment has ``avg_logprob`` below ``AVG_LOGPROB_LIMIT``.
      3. Every segment has ``compression_ratio`` above
         ``COMPRESSION_RATIO_LIMIT`` (repetition loop).
      4. A short transcript dominated by known Whisper boilerplate.

    Args:
        text: The stripped transcript text.
        segments: Segment dicts from a ``verbose_json`` transcription.

    Returns:
        True when the transcript should be discarded as hallucinated.
    """
    if len(text) <= BOILERPLATE_MAX_LENGTH and BOILERPLATE_RE.search(text):
        logger.debug("Transcript rejected (boilerplate): %s", log.snippet(text))
        return True
    if not segments:
        return False
    if all(seg.get("no_speech_prob", 0.0) > NO_SPEECH_PROB_LIMIT for seg in segments):
        logger.debug("Transcript rejected (no_speech_prob): %s", log.snippet(text))
        return True
    if all(seg.get("avg_logprob", 0.0) < AVG_LOGPROB_LIMIT for seg in segments):
        logger.debug("Transcript rejected (avg_logprob): %s", log.snippet(text))
        return True
    if all(seg.get("compression_ratio", 0.0) > COMPRESSION_RATIO_LIMIT for seg in segments):
        logger.debug("Transcript rejected (compression_ratio): %s", log.snippet(text))
        return True
    return False


async def transcribe_bytes(audio_bytes: bytes, media_type: str, filename: str = "") -> str:
    """Transcribe raw audio bytes via Groq Whisper with retries.

    The call is pinned to Russian, uses ``verbose_json`` for segment metadata
    and rejects garbage transcripts (see :func:`is_garbage_transcript`),
    returning ``""`` so the existing empty-transcript handling applies.
    """
    if not filename:
        filename = "voice.ogg" if media_type == "voice" else "video_note.mp4"
    client = AsyncGroq(api_key=config.GROQ_API_KEY, timeout=WHISPER_TIMEOUT)
    last_err: Exception | None = None
    for attempt in range(WHISPER_RETRIES + 1):
        try:
            result = await client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model=config.WHISPER_MODEL,
                language=config.WHISPER_LANGUAGE,
                response_format="verbose_json",
                temperature=0.0,
            )
            text = result.text.strip()
            segments = getattr(result, "segments", None) or []
            if is_garbage_transcript(text, segments):
                return ""
            return text
        except Exception as err:
            last_err = err
            if attempt < WHISPER_RETRIES:
                logger.warning("Transcription attempt %d failed, retrying: %s", attempt + 1, err)
                await asyncio.sleep(2 ** attempt)
    logger.error("Transcription failed after %d attempts: %s", WHISPER_RETRIES + 1, last_err)
    return ""


async def transcribe_voice(file_id: str, media_type: str, bot) -> str:
    """Download a voice or video_note file and return its Whisper transcript."""
    try:
        filename = "voice.ogg" if media_type == "voice" else "video_note.mp4"
        tg_file = await bot.get_file(file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(buffer)
        buffer.seek(0)
        audio_bytes = buffer.read()
    except Exception as err:
        logger.error("Transcription download failed for %s: %s", file_id, err)
        return ""
    return await transcribe_bytes(audio_bytes, media_type, filename)


async def describe_image_bytes(raw_bytes: bytes) -> tuple[bool | None, str]:
    """Describe raw image bytes (PNG/WEBP/JPEG) in Russian via the vision LLM.

    Args:
        raw_bytes: The image payload; the mime type is sniffed from magic
            bytes, defaulting to JPEG.

    Returns:
        ``(is_real_person, description)`` — see :func:`parse_vision_response`.
        Errors propagate to the caller.
    """
    if raw_bytes[:4] == b'\x89PNG':
        mime = "image/png"
    elif raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    b64_image = base64.b64encode(raw_bytes).decode()
    llm = _make_vision_llm()
    response = await ainvoke_with_backoff(llm, [
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
            {"type": "text", "text": VISION_PROMPT},
        ]),
    ])
    return parse_vision_response(response.content)


async def describe_photo(file_id: str, bot) -> tuple[bool | None, str]:
    """Download a Telegram photo and return its real-person classification + description.

    Returns:
        ``(is_real_person, description)`` — see :func:`parse_vision_response`.
        ``(None, "")`` on any download or vision failure.
    """
    try:
        tg_file = await bot.get_file(file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(buffer)
        return await describe_image_bytes(buffer.getvalue())
    except Exception as err:
        logger.error("Photo description failed for file %s: %s", file_id, err)
        return None, ""


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


async def describe_frame(frame_bytes: bytes) -> tuple[bool | None, str]:
    """Describe a single video keyframe via the vision LLM.

    Returns:
        ``(is_real_person, description)`` — see :func:`parse_vision_response`.
    """
    b64_image = base64.b64encode(frame_bytes).decode()
    llm = _make_vision_llm()
    response = await ainvoke_with_backoff(llm, [
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            {"type": "text", "text": VISION_PROMPT},
        ]),
    ])
    return parse_vision_response(response.content)


async def extract_and_describe_frames(video_bytes: bytes) -> list[tuple[bool | None, str]]:
    """Extract keyframes from a video and describe each via the vision LLM.

    Returns:
        ``(is_real_person, description)`` pairs for successfully described
        frames; a failed frame extraction or a failed individual frame
        description is dropped rather than raised.
    """
    loop = asyncio.get_event_loop()
    try:
        frames = await loop.run_in_executor(None, extract_frames_sync, video_bytes)
    except Exception as err:
        logger.warning("Frame extraction failed: %s", err)
        return []
    results = await asyncio.gather(
        *[describe_frame(frame) for frame in frames],
        return_exceptions=True,
    )
    return [
        result for result in results
        if isinstance(result, tuple) and result[1]
    ]


def compose_video_content(transcript: str, frame_descriptions: list[str]) -> str:
    """Join a Whisper transcript and frame descriptions into labelled lines.

    Args:
        transcript: Whisper transcript of the audio track (may be empty).
        frame_descriptions: Vision-LLM descriptions of extracted keyframes.

    Returns:
        Newline-joined labelled block, empty when both inputs are empty.
    """
    parts = []
    if transcript:
        # Without frames (long video or extraction failure) the transcript is
        # just the soundtrack — label it so the response model does not treat
        # lyrics or off-screen speech as the sender's own words.
        if frame_descriptions:
            parts.append(f"[Аудио]: {transcript}")
        else:
            parts.append(
                f"[Аудиодорожка видео — возможно музыка или речь за кадром]: {transcript}"
            )
    total = len(frame_descriptions)
    for index, description in enumerate(frame_descriptions, start=1):
        parts.append(f"[Видео {index}/{total}]: {description}")
    return "\n".join(parts)


async def transcribe_video(file_id: str, media_type: str, bot) -> tuple[bool | None, str]:
    """Download a video/video_note, transcribe audio and describe frames.

    Returns:
        ``(is_real_person, content)`` — ``is_real_person`` aggregates the
        per-frame classifications (see :func:`aggregate_real_person`);
        ``content`` is the composed transcript + frame-description block.
        ``(None, "")`` on a download failure.
    """
    try:
        tg_file = await bot.get_file(file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(buffer)
        video_bytes = buffer.getvalue()
    except Exception as err:
        logger.error("Video download failed for file %s: %s", file_id, err)
        return None, ""
    transcript, frame_results = await asyncio.gather(
        transcribe_bytes(video_bytes, media_type),
        extract_and_describe_frames(video_bytes),
    )
    frame_descriptions = [description for _, description in frame_results]
    is_real_person = aggregate_real_person(frame_results)
    return is_real_person, compose_video_content(transcript, frame_descriptions)


def compose_comments_block(comments: list[dict]) -> str:
    """Render top YouTube comments as a labelled audience-reaction block.

    Args:
        comments: Comment dicts from yt-dlp's info dict (already top-sorted).

    Returns:
        A ``[Топ-комментарии зрителей]`` block with like counts, or ``""``
        when there are no usable comments (missing or disabled).
    """
    lines = []
    for comment in comments[:shorts.MAX_COMMENTS]:
        text = (comment.get("text") or "").strip()
        if not text:
            continue
        if len(text) > shorts.COMMENT_CHAR_LIMIT:
            text = text[:shorts.COMMENT_CHAR_LIMIT] + "…"
        like_count = comment.get("like_count") or 0
        lines.append(f"- ({like_count} лайков) {text}")
    if not lines:
        return ""
    return "\n".join(["[Топ-комментарии зрителей]:", *lines])


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


async def summarize_youtube_short(url: str) -> str:
    """Download a YouTube Short and return its labelled content block.

    Reuses the Telegram-video machinery: Whisper on the audio track and
    vision descriptions of extracted keyframes, plus the top comments as
    audience reaction.

    Args:
        url: Canonical Shorts URL detected by the router.

    Returns:
        Labelled block (header, audio/frames, comments), or ``""`` when the
        download failed or produced neither transcript nor frames — a title
        and comments alone are not enough to react to the video honestly.
    """
    downloaded = await shorts.download_short(url)
    if downloaded is None:
        return ""
    video_bytes, info = downloaded
    transcript, frame_results = await asyncio.gather(
        transcribe_bytes(video_bytes, "video", "short.mp4"),
        extract_and_describe_frames(video_bytes),
    )
    frame_descriptions = [description for _, description in frame_results]
    if not transcript and not frame_descriptions:
        logger.warning("Shorts content extraction produced nothing for %s", url)
        return ""
    if len(transcript) > shorts.TRANSCRIPT_CHAR_LIMIT:
        transcript = transcript[:shorts.TRANSCRIPT_CHAR_LIMIT] + "…"
    parts = [
        compose_short_header(info or {}),
        compose_video_content(transcript, frame_descriptions),
    ]
    comments_block = compose_comments_block((info or {}).get("comments") or [])
    if comments_block:
        parts.append(comments_block)
    return "\n".join(parts)


class MessageIngester:
    """Converts media messages to text and updates the unified_messages store."""

    async def __call__(self, state: BotState) -> dict:
        """Convert the incoming message's media (or Shorts link) to text.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict with the enriched ``incoming`` message, the
            ``media_is_real_person`` classification (photo/video_note/video
            only; None otherwise) and, for Shorts triggers, the
            ``youtube_short_content`` success flag.
        """
        msg = state["incoming"]
        media_type = msg["media_type"]
        bot = state["context_types"].bot
        short_content: str | None = None
        is_real_person: bool | None = None

        if media_type == "text":
            processed, short_content = await self.__ingest_text(state)
        elif media_type == "voice":
            processed = await transcribe_voice(msg["file_id"], "voice", bot)
        elif media_type in ("video_note", "video"):
            is_real_person, processed = await transcribe_video(msg["file_id"], media_type, bot)
        elif media_type == "photo":
            is_real_person, processed = await self.__ingest_photo(msg, bot)
        elif media_type == "sticker" and state["should_respond"]:
            # Only when the bot is about to answer — plain sticker traffic is
            # enriched lazily (reply chains / replied-to lookups) to avoid a
            # vision call per first-seen sticker in the chat.
            processed = await describe_sticker(msg["file_id"], bot)
        else:
            processed = msg["raw_text"] or ""

        if (media_type != "text" or short_content) and processed:
            await self.__update_stored_content(msg, processed)

        incoming_update = dict(state["incoming"])
        incoming_update["processed_text"] = processed
        result: dict = {"incoming": incoming_update, "media_is_real_person": is_real_person}
        if state.get("response_trigger") == "youtube_short":
            result["youtube_short_content"] = short_content or None
        return result

    async def __ingest_photo(self, msg: dict, bot) -> tuple[bool | None, str]:
        """Describe an incoming photo and combine it with its caption.

        Args:
            msg: IncomingMessage dict of the photo message.
            bot: Telegram bot instance used to download the photo.

        Returns:
            ``(is_real_person, processed_text)`` — see :func:`describe_photo`;
            ``processed_text`` falls back to the bare caption when the vision
            call produced no description.
        """
        is_real_person, description = await describe_photo(msg["file_id"], bot)
        caption = msg["raw_text"] or ""
        if description:
            return is_real_person, unified_messages.combine_description_and_caption(description, caption)
        return is_real_person, caption

    async def __update_stored_content(self, msg: dict, content: str) -> None:
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
        except Exception as err:
            logger.warning("Failed to update message content for %s: %s", msg["message_id"], err)

    async def __ingest_text(self, state: BotState) -> tuple[str, str | None]:
        """Process a text message, summarizing a Shorts link when routed so.

        Args:
            state: Current pipeline state.

        Returns:
            Tuple of the processed text (user text, with the labelled video
            block appended on a successful Shorts summary) and the summary
            block itself (``None`` when absent or failed).
        """
        raw_text = state["incoming"]["raw_text"] or ""
        short_url = state.get("youtube_short_url")
        if not short_url:
            return raw_text, None
        short_content = await summarize_youtube_short(short_url)
        if not short_content:
            return raw_text, None
        return f"{raw_text}\n\n{short_content}".strip(), short_content


def extract_frames_sync(video_bytes: bytes) -> list[bytes]:
    buffer = io.BytesIO(video_bytes)
    with av.open(buffer) as container:
        if not container.streams.video:
            return []

        stream = container.streams.video[0]
        duration_seconds = float(container.duration) / 1_000_000 if container.duration else 0

        if duration_seconds > FRAME_DURATION_AUDIO_ONLY:
            return []

        fractions = [0.5] if duration_seconds < FRAME_DURATION_SINGLE else [0.25, 0.5, 0.75]
        frames = []
        for fraction in fractions:
            seek_offset = int(duration_seconds * fraction * 1_000_000)
            container.seek(seek_offset)
            for frame in container.decode(stream):
                img_buffer = io.BytesIO()
                frame.to_image().save(img_buffer, format="JPEG")
                frames.append(img_buffer.getvalue())
                break

    return frames
