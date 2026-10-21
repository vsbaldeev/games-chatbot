"""Whisper transcription of voice and video-note audio.

Transcription is pinned to Russian (``config.WHISPER_LANGUAGE``) and runs a
garbage-transcript check on Whisper's segment metadata plus a boilerplate
denylist: silence/noise hallucinations («Продолжение следует…», «Спасибо за
просмотр») are rejected and treated as empty transcripts."""

import asyncio
import io
import re

from groq import AsyncGroq

from src import config
from src import log

logger = log.get_logger(__name__)


WHISPER_TIMEOUT = 60.0   # seconds; Groq Whisper usually responds in 2–15 s


WHISPER_RETRIES = 2


# Garbage-transcript thresholds — tune from the rejection info-logs.
NO_SPEECH_PROB_LIMIT = 0.6    # above on every segment → no actual speech


AVG_LOGPROB_LIMIT = -1.0      # below on every segment → low-confidence decode


COMPRESSION_RATIO_LIMIT = 2.4  # above on every segment → repetition loop


BOILERPLATE_MAX_LENGTH = 60   # denylist applies only to short transcripts


# Softer companion to AVG_LOGPROB_LIMIT's all()-based hard reject: catches a
# transcript that is only partially garbled (a real sentence decoded through
# background noise) instead of requiring every segment to fail. Not a reject
# threshold — see is_low_confidence_transcript.
LOW_CONFIDENCE_MEAN_LOGPROB = -0.7


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


def is_low_confidence_transcript(segments: list[dict]) -> bool:
    """True when Whisper's mean confidence across segments is low.

    Unlike :func:`is_garbage_transcript`'s ``all()`` checks — built to catch
    total silence, music or a repetition loop — this looks at the *mean*
    ``avg_logprob``, so a transcript that is only partially garbled (a real
    sentence decoded through background noise) is flagged instead of passing
    through as verified speech. Does not replace :func:`is_garbage_transcript`;
    call on transcripts that already passed it.

    Args:
        segments: Segment dicts from a ``verbose_json`` transcription.

    Returns:
        True when the mean ``avg_logprob`` is below ``LOW_CONFIDENCE_MEAN_LOGPROB``.
    """
    if not segments:
        return False
    mean_logprob = sum(seg.get("avg_logprob", 0.0) for seg in segments) / len(segments)
    return mean_logprob < LOW_CONFIDENCE_MEAN_LOGPROB


async def transcribe_bytes(audio_bytes: bytes, media_type: str, filename: str = "") -> tuple[str, bool]:
    """Transcribe raw audio bytes via Groq Whisper with retries.

    The call is pinned to Russian, uses ``verbose_json`` for segment metadata
    and rejects garbage transcripts (see :func:`is_garbage_transcript`),
    returning ``("", False)`` so the existing empty-transcript handling
    applies. A transcript that passes the garbage check but still has low
    mean confidence (see :func:`is_low_confidence_transcript`) is returned
    with its confidence flag set, rather than being trusted as accurate
    speech.

    Returns:
        ``(text, is_low_confidence)``.
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
                return "", False
            return text, is_low_confidence_transcript(segments)
        except Exception as err:
            last_err = err
            if attempt < WHISPER_RETRIES:
                logger.warning("Transcription attempt %d failed, retrying: %s", attempt + 1, err)
                await asyncio.sleep(2 ** attempt)
    logger.error("Transcription failed after %d attempts: %s", WHISPER_RETRIES + 1, last_err)
    return "", False


async def transcribe_voice(file_id: str, media_type: str, bot) -> tuple[str, bool]:
    """Download a voice or video_note file and return its Whisper transcript.

    Returns:
        ``(text, is_low_confidence)`` — see :func:`transcribe_bytes`.
    """
    try:
        filename = "voice.ogg" if media_type == "voice" else "video_note.mp4"
        tg_file = await bot.get_file(file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(buffer)
        buffer.seek(0)
        audio_bytes = buffer.read()
    except Exception as err:
        logger.error("Transcription download failed for %s: %s", file_id, err)
        return "", False
    return await transcribe_bytes(audio_bytes, media_type, filename)
