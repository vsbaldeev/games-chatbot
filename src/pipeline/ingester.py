"""
MessageIngester — second node in the LangGraph pipeline.

Processes raw media into readable text:
  - text       → passed through unchanged
  - voice      → transcribed via Groq Whisper
  - video_note → transcribed via Groq Whisper + frames described via Vision LLM
  - video      → transcribed via Groq Whisper + frames described via Vision LLM
  - photo      → described via vision LLM (one-sentence description)

Frame extraction (PyAV):
  duration < 15s   → 1 keyframe (middle)
  15s – 120s       → 3 keyframes (uniformly distributed)
  > 120s           → audio only, no frames

Updates the unified_messages row written by the Router with the real content
so that reply-chain queries later in the pipeline return useful text.
"""

import asyncio
import base64
import io
from src import log

import av
from groq import AsyncGroq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src import config
from src.pipeline.state import BotState
from src.store import unified_messages

logger = log.get_logger(__name__)

WHISPER_MODEL = "whisper-large-v3"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

VISION_PROMPT = (
    "Опиши изображение кратко по-русски (1–2 предложения). "
    "Если на изображении узнаваемый человек — назови его имя (знаменитость, актёр, стример, спортсмен, политик и др.). "
    "Если это скриншот, арт или интерфейс из игры, фильма, сериала или аниме — назови конкретное название. "
    "Упомяни любой видимый текст, никнеймы или оверлеи, которые помогают определить, кто или что изображено."
)

FRAME_DURATION_AUDIO_ONLY = 120
FRAME_DURATION_SINGLE = 15

WHISPER_TIMEOUT = 60.0   # seconds; Groq Whisper usually responds in 2–15 s
WHISPER_RETRIES = 2


async def transcribe_bytes(audio_bytes: bytes, media_type: str, filename: str = "") -> str:
    """Transcribe raw audio bytes via Groq Whisper with retries."""
    if not filename:
        filename = "voice.ogg" if media_type == "voice" else "video_note.mp4"
    client = AsyncGroq(api_key=config.GROQ_API_KEY, timeout=WHISPER_TIMEOUT)
    last_err: Exception | None = None
    for attempt in range(WHISPER_RETRIES + 1):
        try:
            result = await client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model=WHISPER_MODEL,
            )
            return result.text.strip()
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


async def describe_photo(file_id: str, bot) -> str:
    """Download a Telegram photo and return a one-sentence Russian description via vision LLM."""
    try:
        tg_file = await bot.get_file(file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(buffer)
        raw_bytes = buffer.getvalue()

        if raw_bytes[:4] == b'\x89PNG':
            mime = "image/png"
        elif raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
            mime = "image/webp"
        else:
            mime = "image/jpeg"

        b64_image = base64.b64encode(raw_bytes).decode()
        llm = ChatGroq(model=VISION_MODEL, api_key=config.GROQ_API_KEY, temperature=0.1, max_tokens=200, max_retries=0)
        response = await llm.ainvoke([
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
                {"type": "text", "text": VISION_PROMPT},
            ]),
        ])
        return response.content.strip()
    except Exception as err:
        logger.error("Photo description failed for file %s: %s", file_id, err)
        return ""


async def describe_frame(frame_bytes: bytes) -> str:
    b64_image = base64.b64encode(frame_bytes).decode()
    llm = ChatGroq(model=VISION_MODEL, api_key=config.GROQ_API_KEY, temperature=0.1, max_tokens=200, max_retries=0)
    response = await llm.ainvoke([
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            {"type": "text", "text": VISION_PROMPT},
        ]),
    ])
    return response.content.strip()


async def extract_and_describe_frames(video_bytes: bytes) -> list[str]:
    loop = asyncio.get_event_loop()
    try:
        frames = await loop.run_in_executor(None, extract_frames_sync, video_bytes)
    except Exception as err:
        logger.warning("Frame extraction failed: %s", err)
        return []
    descriptions = await asyncio.gather(
        *[describe_frame(frame) for frame in frames],
        return_exceptions=True,
    )
    return [desc for desc in descriptions if isinstance(desc, str) and desc]


async def transcribe_video(file_id: str, media_type: str, bot) -> str:
    """Download a video/video_note, transcribe audio and describe frames."""
    try:
        tg_file = await bot.get_file(file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(buffer)
        video_bytes = buffer.getvalue()
    except Exception as err:
        logger.error("Video download failed for file %s: %s", file_id, err)
        return ""
    transcript, frame_descriptions = await asyncio.gather(
        transcribe_bytes(video_bytes, media_type),
        extract_and_describe_frames(video_bytes),
    )
    parts = []
    if transcript:
        parts.append(f"[Аудио]: {transcript}")
    total = len(frame_descriptions)
    for index, description in enumerate(frame_descriptions, start=1):
        parts.append(f"[Видео {index}/{total}]: {description}")
    return "\n".join(parts)


class MessageIngester:
    """Converts media messages to text and updates the unified_messages store."""

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        media_type = msg["media_type"]
        bot = state["context_types"].bot

        if media_type == "text":
            processed = msg["raw_text"] or ""
        elif media_type == "voice":
            processed = await transcribe_voice(msg["file_id"], "voice", bot)
        elif media_type in ("video_note", "video"):
            processed = await transcribe_video(msg["file_id"], media_type, bot)
        elif media_type == "photo":
            description = await describe_photo(msg["file_id"], bot)
            caption = msg["raw_text"] or ""
            if description:
                processed = unified_messages.combine_description_and_caption(description, caption)
            else:
                processed = caption
        else:
            processed = msg["raw_text"] or ""

        if media_type != "text" and processed:
            try:
                await unified_messages.update_content(
                    chat_id=msg["chat_id"],
                    message_id=msg["message_id"],
                    content=processed,
                )
            except Exception as err:
                logger.warning("Failed to update message content for %s: %s", msg["message_id"], err)

        incoming_update = dict(state["incoming"])
        incoming_update["processed_text"] = processed
        return {"incoming": incoming_update}


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
