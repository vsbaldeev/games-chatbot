"""Correction classifier: one small Groq call deciding whether a direct reply
to a bot message disputes or corrects a factual claim it made.

Mirrors src/memes/judge.py's shape (one cheap classification call, fail
closed on any error or unparseable output) but for text rather than vision,
and TAG_MODEL rather than the vision model — same reasoning-model call
convention used by src/jobs/roles.py's call_role_model (reasoning_effort="none").
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent import ainvoke_with_backoff
from src.config.prompts import CORRECTION_CLASSIFIER_SYSTEM

logger = log.get_logger(__name__)

MAX_TOKENS = 10  # the verdict is a single word


async def classify_correction(bot_text: str, reply_text: str) -> bool:
    """Decide whether `reply_text` corrects a factual claim in `bot_text`.

    Args:
        bot_text: The bot message being replied to.
        reply_text: The direct reply to classify.

    Returns:
        True for a CORRECTION verdict; False for OTHER, an unparseable
        response, or any classifier failure — fails closed, since an
        uncertain reply must not be counted against the bot.
    """
    try:
        llm = ChatGroq(
            model=config.TAG_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0,
            max_tokens=MAX_TOKENS,
            max_retries=0,
            reasoning_effort="none",
        )
        user_content = f"Bot message: {bot_text}\nReply: {reply_text}"
        response = await ainvoke_with_backoff(llm, [
            SystemMessage(content=CORRECTION_CLASSIFIER_SYSTEM),
            HumanMessage(content=user_content),
        ])
        verdict = (response.content or "").strip().upper()
        return "CORRECTION" in verdict
    except Exception as err:
        logger.warning("Correction classification failed, failing closed: %s", err)
        return False
