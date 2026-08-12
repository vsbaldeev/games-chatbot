"""Vision-LLM gate deciding whether a scraped candidate is really a meme.

The Telegram sources scrape public channel web previews, where a channel
author's own post — a donation appeal, an ad, an "I'm back" announcement —
carries a standalone photo in exactly the same markup as a meme. No parser can
separate the two, so the image itself is scored here before it reaches a chat.

The same score covers a second requirement: captions are never reposted, so a
photo that is only funny under the source channel's caption must not ship
either. See :data:`src.config.prompts.MEME_JUDGE_SYSTEM`.
"""

import base64

from langchain_core.messages import HumanMessage, SystemMessage

from src import config, log
from src.agent.vision import make_vision_llm
from src.config.prompts import MEME_JUDGE_SYSTEM
from src.utils.llm_json import load_json_object

logger = log.get_logger(__name__)

MAGIC_BYTE_MIMES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
DEFAULT_MIME = "image/jpeg"


def detect_image_mime(image_bytes: bytes) -> str:
    """Identify an image's MIME type from its magic bytes.

    Meme CDNs serve JPEG, PNG and WebP interchangeably and the data URL must
    declare the format it actually carries, so the type is sniffed rather than
    assumed (unlike :mod:`src.life.photo_judge`, whose input is always PNG from
    the local image generator).

    Args:
        image_bytes: Raw downloaded image.

    Returns:
        An image MIME type, falling back to ``image/jpeg`` for unrecognised
        data — the overwhelmingly common case among these sources.
    """
    for magic, mime in MAGIC_BYTE_MIMES:
        if image_bytes.startswith(magic):
            return mime
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return DEFAULT_MIME


def parse_verdict(data: dict | None) -> int | None:
    """Extract and validate the 0-10 score from the judge's JSON verdict.

    Args:
        data: Parsed JSON object from the model, or None.

    Returns:
        The score, or None when it is missing, non-numeric or out of range.
    """
    if data is None:
        return None
    raw_score = data.get("score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return None
    score = int(raw_score)
    if score < 0 or score > 10:
        return None
    return score


async def score_meme(image_bytes: bytes) -> int | None:
    """Score how well an image works as a captionless meme in a group chat.

    Args:
        image_bytes: The downloaded candidate image.

    Returns:
        Score 0-10, or None when the judge is unavailable or answered
        unusably — "unknown", never zero: an outage must not be recorded as a
        verdict against the candidate.
    """
    try:
        b64_image = base64.b64encode(image_bytes).decode()
        mime = detect_image_mime(image_bytes)
        response = await make_vision_llm(max_tokens=config.MEME_JUDGE_MAX_TOKENS).ainvoke([
            SystemMessage(content=MEME_JUDGE_SYSTEM),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
            ]),
        ])
        score = parse_verdict(load_json_object(response.content or "", context="Meme judging"))
        logger.info("Meme judge scored %s (pass >= %s)", score, config.MEME_JUDGE_PASS_SCORE)
        return score
    except Exception as error:
        logger.warning("Meme judging failed: %s", error)
        return None
