"""Cross-provider vision LLM factory shared by every vision-LLM call site.

Independent call sites (ingester frame/photo description, meme vetting) each
built their own single-provider client, so one provider's quota exhaustion,
deprecation or outage degraded every one of them silently. This mirrors
src.pipeline.ingest.filter.classify.make_filter_llm's fallback for the text
filter, with OpenRouter as the primary and Groq as the reserve.
"""

from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import APIError as OpenRouterAPIError

from src import config, log

logger = log.get_logger(__name__)


def make_vision_llm(max_tokens: int) -> Runnable:
    """Build a vision LLM: OpenRouter primary, Groq reserve.

    VISION_PRIMARY_MODEL and VISION_RESERVE_MODEL are both reasoning models;
    each leg switches reasoning off in its provider's own way (OpenRouter's
    ``reasoning`` request field, Groq's ``reasoning_effort``), because
    thinking would otherwise eat the whole token budget and return an empty
    description. Any openai.APIError from the primary (404 for a deprecated
    model, 429s, connection errors, 5xx) fails over to the Groq reserve.
    Without OPENROUTER_API_KEY, the Groq client is returned bare and callers
    degrade on their own, the same fail-open contract make_filter_llm uses
    for the text filter.

    The primary carries max_retries=2: its 429s come from OpenRouter's
    shared pool throttling short-term, so a couple of backoff retries
    (handled by the underlying OpenAI SDK client) can clear it instead of
    dropping to the reserve. The Groq reserve gets none: its 429s are
    daily-quota exhaustion a same-model retry can never recover from.

    Args:
        max_tokens: Response token budget of the calling site.

    Returns:
        A Runnable accepting a message list and returning one response.
    """
    reserve_llm = ChatGroq(
        model=config.VISION_RESERVE_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=max_tokens,
        max_retries=0,
        reasoning_effort="none",
    )
    if not config.OPENROUTER_API_KEY:
        logger.warning(
            "Vision: OPENROUTER_API_KEY unset — %s is the only vision model", config.VISION_RESERVE_MODEL,
        )
        return reserve_llm
    primary_llm = ChatOpenAI(
        model=config.VISION_PRIMARY_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=0.1,
        max_tokens=max_tokens,
        max_retries=2,
        extra_body={"reasoning": {"enabled": False}},
    )
    return primary_llm.with_fallbacks([reserve_llm], exceptions_to_handle=(OpenRouterAPIError,))
