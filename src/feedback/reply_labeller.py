"""Batched reply labelling: one OpenRouter call labels up to a batch of
replies with a reaction and, for negative reactions, a cause.

OpenRouter-only by design (see REPLY_LABEL_MODELS): this job is offline and
batched, so a rate limit costs a delay rather than a lost measurement, and it
must never compete with chat replies for Groq quota.

Parsing is strict and item-wise. A malformed item is dropped so its row stays
pending and gets retried; a batch-level problem (duplicate ids, non-array
JSON) drops everything, for the same reason. Nothing is ever guessed — a
wrong label is worse than a missing one, because a missing one is visible in
the unlabelled count.

Two failure modes, and the caller must be able to tell them apart. A model
that answered but got an item wrong is reported by that item's absence from
the returned dict: the row's own content is what the model choked on, so the
row has earned a retry attempt. A call that never reached a model at all —
no API key, a transport error, every fallback leg down — raises
LabellingCallFailed instead, because an OpenRouter outage says nothing about
any particular row and must not spend its attempts. Without that split a
45-minute outage (three runs at the 15-minute schedule) would retire every
queued reply as UNCLEAR, backfill included.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src import config, log
from src.agent import ainvoke_with_backoff
from src.config.prompts import REPLY_LABEL_SYSTEM

logger = log.get_logger(__name__)

# MEDIA is deliberately absent: it is a rule label applied by the job to
# wordless replies. Accepting it here would let a model that found a text reply
# hard to read file it as "media", which is a lie about the reply's shape.
MODEL_REACTIONS: frozenset[str] = frozenset({
    "POSITIVE", "NEGATIVE", "NEUTRAL", "UNCLEAR",
})
CAUSES: frozenset[str] = frozenset({
    "PERSONA", "INVENTED_LIFE", "FACTUAL", "TONE", "IRRELEVANT", "UNCLEAR",
})
CAUSE_BEARING_REACTION = "NEGATIVE"

ITEM_KEYS = frozenset({"id", "reaction", "cause"})


class LabellingCallFailed(Exception):
    """The labelling call never produced a response to parse.

    Raised for a global failure — a missing OPENROUTER_API_KEY, a transport
    error, every fallback leg refusing — as opposed to a response that came
    back and was wrong about some item. The job treats the two differently:
    only the second kind costs a row one of its three attempts.
    """


def build_batch_payload(rows: list[dict]) -> str:
    """Render pending rows as the JSON array the prompt describes.

    Args:
        rows: fetch_pending dicts.

    Returns:
        A JSON string, one object per row, with ensure_ascii off so Russian
        text reaches the model as text rather than escapes.
    """
    items = [
        {
            "id": row["reply_message_id"],
            "bot_message": row["bot_content"],
            "source": row["source"],
            "trigger": row["trigger"],
            "reply": row["reply_content"],
        }
        for row in rows
    ]
    return json.dumps(items, ensure_ascii=False)


def strip_code_fence(raw: str) -> str:
    """Remove a surrounding markdown fence, which some free-tier models add."""
    text = raw.strip()
    if not text.startswith("```"):
        return text
    without_open = text.split("\n", 1)[-1]
    return without_open.rsplit("```", 1)[0].strip()


def read_item(item: object, expected_ids: set[int]) -> tuple[int, str, str | None] | None:
    """Validate one response item, or return None to drop it.

    A dropped item leaves its row pending, so the next run retries it.
    """
    if not isinstance(item, dict) or set(item.keys()) != ITEM_KEYS:
        return None
    item_id = item["id"]
    reaction = item["reaction"]
    if not isinstance(item_id, int) or isinstance(item_id, bool):
        return None
    if item_id not in expected_ids or reaction not in MODEL_REACTIONS:
        return None
    if reaction != CAUSE_BEARING_REACTION:
        return item_id, reaction, None
    cause = item["cause"]
    return item_id, reaction, cause if cause in CAUSES else "UNCLEAR"


def parse_label_batch(raw: str, expected_ids: set[int]) -> dict[int, tuple[str, str | None]]:
    """Parse a batch response into {reply_message_id: (reaction, cause)}.

    Args:
        raw: The model's raw output.
        expected_ids: The reply ids sent in this batch.

    Returns:
        Only the items that validated. An empty dict means nothing in the
        response could be trusted — every row stays pending.
    """
    try:
        parsed = json.loads(strip_code_fence(raw))
    except (json.JSONDecodeError, TypeError):
        logger.warning("Reply labelling: response was not JSON, dropping batch")
        return {}
    if not isinstance(parsed, list):
        logger.warning("Reply labelling: response was not a JSON array, dropping batch")
        return {}
    labels: dict[int, tuple[str, str | None]] = {}
    for item in parsed:
        read = read_item(item, expected_ids)
        if read is None:
            continue
        item_id, reaction, cause = read
        if item_id in labels:
            logger.warning("Reply labelling: duplicate id %s, dropping batch", item_id)
            return {}
        labels[item_id] = (reaction, cause)
    return labels


def make_labeller_llm(max_tokens: int) -> ChatOpenAI:
    """Build the OpenRouter client, primary leg with the second as fallback.

    ``reasoning_effort="none"`` is the same idiom make_filter_llm
    (src/pipeline/filter_node.py) and the vision/roles calls already use in
    this repo for the same class of problem: the primary leg
    (nvidia/nemotron-3-super-120b-a12b:free) is a reasoning model with
    reasoning ON by default, and a <think> block would eat the whole
    REPLY_LABEL_MAX_TOKENS_PER_ITEM budget before any JSON appeared. Measured
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


async def label_batch(rows: list[dict]) -> dict[int, tuple[str, str | None]]:
    """Label one batch of pending rows.

    Fails closed either way — no label is ever guessed — but the two ways are
    distinguishable, because only one of them is the row's own fault.

    Returns:
        {reply_message_id: (reaction, cause)} for every row the model
        labelled cleanly. A row absent from the result is one the model
        answered about wrongly or not at all, which is what earns it an
        attempt.

    Raises:
        LabellingCallFailed: The call never produced a response — no API key,
            a transport error, or every fallback leg failing. Nothing was
            learned about any row in the batch, so the caller must leave
            their attempts alone.
    """
    if not rows:
        return {}
    if not config.OPENROUTER_API_KEY:
        raise LabellingCallFailed("OPENROUTER_API_KEY unset")
    expected_ids = {row["reply_message_id"] for row in rows}
    try:
        llm = make_labeller_llm(len(rows) * config.REPLY_LABEL_MAX_TOKENS_PER_ITEM)
        response = await ainvoke_with_backoff(llm, [
            SystemMessage(content=REPLY_LABEL_SYSTEM),
            HumanMessage(content=build_batch_payload(rows)),
        ])
    except Exception as error:
        raise LabellingCallFailed(str(error) or type(error).__name__) from error
    return parse_label_batch(response.content or "", expected_ids)
