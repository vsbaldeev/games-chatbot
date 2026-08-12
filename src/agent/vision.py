"""Cross-provider vision LLM factory shared by every vision-LLM call site.

VISION_MODEL is the only non-deprecated vision-capable model on Groq's free
tier, and three independent call sites (ingester frame/photo description,
meme vetting, selfie candidate scoring) each built their own Groq-only
client for it — so a Groq daily-quota exhaustion or outage degraded every
one of them silently, with no cross-provider recovery. This mirrors
src.pipeline.filter_node.make_filter_llm's fallback for the text filter.
"""

from groq import APIError
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src import config, log

logger = log.get_logger(__name__)


def make_vision_llm(max_tokens: int) -> Runnable:
    """Build a vision LLM with a cross-provider fallback.

    VISION_MODEL is a Groq reasoning model; reasoning_effort="none" keeps it
    from burning the whole token budget inside a <think> block. Any
    groq.APIError from the primary (rate limits, daily-quota 429s a
    same-model retry can never recover from, connection errors, 5xx) fails
    over to an OpenRouter-hosted instruct-mode Qwen3-VL model, which needs no
    such workaround. Without OPENROUTER_API_KEY, the Groq client is returned
    bare and callers degrade on their own, the same fail-open contract
    make_filter_llm uses for the text filter.

    Args:
        max_tokens: Response token budget of the calling site.

    Returns:
        A Runnable accepting a message list and returning one response.
    """
    primary_llm = ChatGroq(
        model=config.VISION_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=max_tokens,
        max_retries=0,
        reasoning_effort="none",
    )
    if not config.OPENROUTER_API_KEY:
        logger.warning("Vision: OPENROUTER_API_KEY unset — no fallback for %s", config.VISION_MODEL)
        return primary_llm
    fallback_llm = ChatOpenAI(
        model=config.VISION_FALLBACK_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=0.1,
        max_tokens=max_tokens,
        max_retries=0,
    )
    return primary_llm.with_fallbacks([fallback_llm], exceptions_to_handle=(APIError,))
