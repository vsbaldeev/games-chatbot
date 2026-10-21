"""Video and video-note handling: keyframe extraction (PyAV) and description.

Frame extraction:
  duration < 15s   → 1 keyframe (middle)
  15s – 120s       → 3 keyframes (uniformly distributed)
  > 120s           → audio only, no frames"""

import asyncio
import base64
import io

import av
from langchain_core.messages import HumanMessage

from src import log
from src.config.prompts import VISION_PROMPT
from src.llm.vision import make_vision_llm
from src.pipeline.ingest.transcription import transcribe_bytes
from src.pipeline.ingest.vision import aggregate_real_person, parse_vision_response

logger = log.get_logger(__name__)


FRAME_DURATION_AUDIO_ONLY = 120


FRAME_DURATION_SINGLE = 15


async def describe_frame(frame_bytes: bytes) -> tuple[bool | None, str]:
    """Describe a single video keyframe via the vision LLM.

    Returns:
        ``(is_real_person, description)`` — see :func:`parse_vision_response`.
    """
    b64_image = base64.b64encode(frame_bytes).decode()
    llm = make_vision_llm(max_tokens=200)
    response = await llm.ainvoke([
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
        description is logged and dropped rather than raised.
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
    described = []
    for result in results:
        if not isinstance(result, tuple):
            logger.warning("Frame description failed: %s", result)
            continue
        if result[1]:
            described.append(result)
    return described


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
    transcribe_result, frame_results = await asyncio.gather(
        transcribe_bytes(video_bytes, media_type),
        extract_and_describe_frames(video_bytes),
    )
    # unused here — video framing doesn't consume the confidence signal
    # (voice-only, see build_voice_trigger_line)
    transcript, transcript_low_confidence = transcribe_result
    frame_descriptions = [description for _, description in frame_results]
    is_real_person = aggregate_real_person(frame_results)
    return is_real_person, compose_video_content(transcript, frame_descriptions)


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
