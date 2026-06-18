"""Language detection and correction utilities."""

import re

from langchain_core.messages import HumanMessage

from src import log
from src.agent.middleware import strip_thinking

logger = log.get_logger(__name__)

FOREIGN_SCRIPT_RE = re.compile(
    "[一-鿿"   # CJK Unified Ideographs
    "㐀-䶿"    # CJK Extension A
    "가-힯"    # Hangul Syllables
    "ᄀ-ᇿ"    # Hangul Jamo
    "぀-ヿ"    # Hiragana + Katakana
    "฀-๿"    # Thai
    "؀-ۿ"    # Arabic
    "֐-׿]"   # Hebrew
)

LANGUAGE_CORRECTION_PROMPT = (
    "Твой предыдущий ответ содержал символы не на русском языке. "
    "Ответь ТОЛЬКО на русском языке."
)


async def apply_language_correction(llm, ai_message, messages: list):
    """Retry the LLM call in Russian if the response contains foreign script.

    Args:
        llm: Language model with an async ``ainvoke`` method.
        ai_message: Original AI response to inspect.
        messages: Full message history used for the correction call.

    Returns:
        Corrected AI message if foreign script was detected, otherwise the original.
    """
    visible = strip_thinking(ai_message.content or "")
    if not visible or not FOREIGN_SCRIPT_RE.search(visible):
        return ai_message
    logger.warning("Foreign script detected, retrying in Russian")
    correction = messages + [HumanMessage(content=LANGUAGE_CORRECTION_PROMPT)]
    try:
        return await llm.ainvoke(correction)
    except Exception as err:
        logger.warning("Language correction failed: %s", err, exc_info=True)
        return ai_message
