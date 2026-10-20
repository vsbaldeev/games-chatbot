"""Single-reply labelling: one OpenRouter call labels one reply to a bot
message as POSITIVE, NEGATIVE, NEUTRAL or UNCLEAR.

OpenRouter-only by design (see REPLY_LABEL_MODELS): the labeller must never
compete with chat replies for Groq quota.

Parsing is strict. Anything outside the contract returns None and the caller
counts the reply as unlabelled — a wrong label is worse than a missing one,
because a missing one is visible in unlabelled_replies.

Two failure modes stay distinguishable: a model that answered badly returns
None; a call that never reached a model (no API key, a transport error, every
fallback leg down) raises LabellingCallFailed.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src import config, log
from src.llm import ainvoke_with_backoff
from src.config.prompts import REPLY_LABEL_SYSTEM

logger = log.get_logger(__name__)

# MEDIA is deliberately absent: wordless replies are counted by rule before
# the model is called, so a model answering MEDIA is out of contract.
MODEL_REACTIONS: frozenset[str] = frozenset({
    "POSITIVE", "NEGATIVE", "NEUTRAL", "UNCLEAR",
})

RESPONSE_KEYS = frozenset({"reaction"})


class LabellingCallFailed(Exception):
    """The labelling call never produced a response to parse.

    Raised for a global failure — a missing OPENROUTER_API_KEY, a transport
    error, every fallback leg refusing — as opposed to a response that came
    back and was wrong.
    """


def build_payload(*, bot_message: str, reply: str, source: str) -> str:
    """Render one reply and its bot message as the JSON object the prompt describes.

    ensure_ascii is off so Russian text reaches the model as text rather than escapes.
    """
    return json.dumps(
        {"bot_message": bot_message, "source": source, "reply": reply},
        ensure_ascii=False,
    )


def strip_code_fence(raw: str) -> str:
    """Remove a surrounding markdown fence, which some free-tier models add."""
    text = raw.strip()
    if not text.startswith("```"):
        return text
    without_open = text.split("\n", 1)[-1]
    return without_open.rsplit("```", 1)[0].strip()


def parse_label(raw: str) -> str | None:
    """Parse the model's answer into one label, or None if it breaks the contract."""
    if not isinstance(raw, str):
        logger.warning("Reply labelling: response content was not text")
        return None
    try:
        parsed = json.loads(strip_code_fence(raw))
    except (json.JSONDecodeError, TypeError):
        logger.warning("Reply labelling: response was not JSON")
        return None
    if not isinstance(parsed, dict) or set(parsed.keys()) != RESPONSE_KEYS:
        logger.warning("Reply labelling: response had the wrong shape")
        return None
    reaction = parsed["reaction"]
    return reaction if isinstance(reaction, str) and reaction in MODEL_REACTIONS else None


def make_labeller_llm(max_tokens: int) -> ChatOpenAI:
    """Build the OpenRouter client, primary leg with the second as fallback.

    ``reasoning_effort="none"`` is the same idiom make_filter_llm
    (src/pipeline/filter_node.py) and the vision/roles calls already use in
    this repo for the same class of problem: the primary leg
    (nvidia/nemotron-3-super-120b-a12b:free) is a reasoning model with
    reasoning ON by default, and a <think> block would eat the whole
    REPLY_LABEL_MAX_TOKENS budget before any JSON appeared. Measured
    on this call's real shape: "none" spends 0 reasoning tokens, "low" spent
    293 and truncated the array. The Gemma fallback leg does not reason by
    default and OpenRouter drops the parameter for models that ignore it.
    """
    legs = [
        ChatOpenAI(
            model=model,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=max_tokens,
            max_retries=0,
            reasoning_effort="none",
        )
        for model in config.REPLY_LABEL_MODELS
    ]
    return legs[0].with_fallbacks(legs[1:]) if len(legs) > 1 else legs[0]


async def label_reply(*, bot_message: str, reply: str, source: str) -> str | None:
    """Label one reply.

    Returns:
        One of MODEL_REACTIONS, or None when the model answered outside the contract.

    Raises:
        LabellingCallFailed: The call never produced a response.
    """
    if not config.OPENROUTER_API_KEY:
        raise LabellingCallFailed("OPENROUTER_API_KEY unset")
    try:
        llm = make_labeller_llm(config.REPLY_LABEL_MAX_TOKENS)
        response = await ainvoke_with_backoff(llm, [
            SystemMessage(content=REPLY_LABEL_SYSTEM),
            HumanMessage(content=build_payload(bot_message=bot_message, reply=reply, source=source)),
        ])
    except Exception as error:
        raise LabellingCallFailed(str(error) or type(error).__name__) from error
    return parse_label(response.content or "")
