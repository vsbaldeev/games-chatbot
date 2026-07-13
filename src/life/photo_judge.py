"""Vision-LLM judge for photo life-post candidates.

SD1.5 renders subject *interactions* stochastically — the same prompt
sometimes yields the described action, sometimes only the subjects standing
around. The poster generates several candidates and this judge scores each
against the episode's ``image_prompt`` (0-10, interaction-weighted) so the
most faithful one ships. It ranks, it does not gate: the poster still posts
the best candidate even when none reaches the pass score.
"""

import base64

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent.middleware import ainvoke_with_backoff
from src.config.prompts import PHOTO_JUDGE_SYSTEM
from src.utils.llm_json import load_json_object

logger = log.get_logger(__name__)


def make_judge_llm() -> ChatGroq:
    """Return a ChatGroq instance configured for photo judging.

    VISION_MODEL is a reasoning model; without ``reasoning_effort="none"``
    the whole token budget burns inside a ``<think>`` block and no JSON
    verdict is produced (same trap as ``src/pipeline/ingester.py``).

    Returns:
        Configured vision LLM.
    """
    return ChatGroq(
        model=config.VISION_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=config.PHOTO_JUDGE_MAX_TOKENS,
        max_retries=0,
        reasoning_effort="none",
    )


def parse_score(data: dict | None) -> int | None:
    """Extract and validate the 0-10 score from the judge's JSON verdict.

    Args:
        data: Parsed JSON object from the model, or None.

    Returns:
        The score clamped to sanity (must already be within 0-10), or None
        when missing, non-numeric or out of range.
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


async def score_photo(png_bytes: bytes, image_prompt: str) -> int | None:
    """Score how faithfully a generated photo depicts the episode's scene.

    Args:
        png_bytes: The candidate image (PNG).
        image_prompt: The episode's English scene description the image was
            generated from.

    Returns:
        Score 0-10 (higher is more faithful, interaction weighted heaviest),
        or None on any failure — "unknown", never zero: a judge outage must
        not make a valid candidate look like a mismatch.
    """
    try:
        b64_image = base64.b64encode(png_bytes).decode()
        response = await ainvoke_with_backoff(make_judge_llm(), [
            SystemMessage(content=PHOTO_JUDGE_SYSTEM),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text", "text": f"Requested scene: {image_prompt}"},
            ]),
        ])
        data = load_json_object(response.content or "", context="Photo judging")
        return parse_score(data)
    except Exception as error:
        logger.warning("Photo judging failed: %s", error)
        return None
