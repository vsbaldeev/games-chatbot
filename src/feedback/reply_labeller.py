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
    """Build the OpenRouter client, primary leg with the second as fallback."""
    legs = [
        ChatOpenAI(
            model=model,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=max_tokens,
            max_retries=0,
        )
        for model in config.REPLY_LABEL_MODELS
    ]
    return legs[0].with_fallbacks(legs[1:]) if len(legs) > 1 else legs[0]


async def label_batch(rows: list[dict]) -> dict[int, tuple[str, str | None]]:
    """Label one batch of pending rows.

    Returns:
        {reply_message_id: (reaction, cause)} for every row that labelled
        cleanly. Rows absent from the result stay pending. An empty dict on
        any failure — fails closed, same stance as the classifier this
        replaces.
    """
    if not rows or not config.OPENROUTER_API_KEY:
        if not config.OPENROUTER_API_KEY:
            logger.warning("Reply labelling: OPENROUTER_API_KEY unset, skipping")
        return {}
    expected_ids = {row["reply_message_id"] for row in rows}
    try:
        llm = make_labeller_llm(len(rows) * config.REPLY_LABEL_MAX_TOKENS_PER_ITEM)
        response = await ainvoke_with_backoff(llm, [
            SystemMessage(content=REPLY_LABEL_SYSTEM),
            HumanMessage(content=build_batch_payload(rows)),
        ])
    except Exception as error:
        logger.warning("Reply labelling call failed, leaving rows pending: %s", error)
        return {}
    return parse_label_batch(response.content or "", expected_ids)
