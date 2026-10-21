"""Vision-LLM descriptions of photos, and the real-person-vs-meme tag.

Every vision call also returns a real-person-vs-meme classification (see
``parse_vision_response``), piggybacked on the same call — no extra LLM round
trip. It is surfaced as ``media_is_real_person`` in the pipeline state; nothing
reads it any more (the meme gate that did was removed)."""

import base64
import io
import re

from langchain_core.messages import HumanMessage

from src import log
from src.config.prompts import VISION_MEME_TAG, VISION_PROMPT, VISION_REAL_PERSON_TAG
from src.llm.vision import make_vision_llm

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
    llm = make_vision_llm(max_tokens=200)
    response = await llm.ainvoke([
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
