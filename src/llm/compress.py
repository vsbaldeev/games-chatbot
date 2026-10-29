"""Caption compression — fit an over-budget link teaser into its budget.

The link caption's teaser is capped at ``HOOK_CHAR_LIMIT`` (see
src/events/link_repost.py). The response prompt already asks for a shorter
line, so this runs rarely; when it does, the teaser is rewritten shorter
rather than cut, so it never ends mid-sentence.

Best effort by contract: any failure returns the input unchanged and the
caller falls back to deterministic sentence-boundary truncation.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import config, log
from src.llm.middleware import ainvoke_with_backoff, strip_thinking
from src.config.prompts import CAPTION_COMPRESS_SYSTEM

logger = log.get_logger(__name__)


async def compress_to_budget(text: str, budget: int) -> str:
    """Rewrite ``text`` shorter so it fits ``budget`` characters.

    Args:
        text: The summary that overflowed the caption limit.
        budget: Character ceiling the rewrite should respect.

    Returns:
        The compressed text, or ``text`` unchanged on any LLM error or empty
        output. The result is not guaranteed to be within ``budget`` — the
        model may ignore it — so callers must still enforce the ceiling.
    """
    llm = ChatGroq(
        model=config.CAPTION_COMPRESS_MODEL, api_key=config.GROQ_API_KEY,
        temperature=0.2, max_tokens=config.CAPTION_COMPRESS_MAX_TOKENS, max_retries=0,
    )
    try:
        result = await ainvoke_with_backoff(llm, [
            SystemMessage(content=CAPTION_COMPRESS_SYSTEM.format(budget=budget)),
            HumanMessage(content=text),
        ])
    except Exception as error:
        logger.warning("Caption compression failed: %s", error)
        return text
    compressed = strip_thinking(result.content or "").strip()
    if not compressed:
        logger.warning("Caption compression returned empty output")
        return text
    return compressed
