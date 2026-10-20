"""Single-response labelling: one OpenRouter call says whether a delivered
reply answered the question it was given — ANSWERED, REFUSED, NON_ANSWER or
UNCLEAR.

Reuses the reply labeller's OpenRouter legs and failure contract on purpose
(see REPLY_LABEL_MODELS): the labeller must never compete with chat replies
for Groq quota, and a call that never reached a model raises
LabellingCallFailed while a model that answered badly returns None. The
caller stores both as UNCLEAR — a missing label would be indistinguishable
from "not labelled yet".

Parsing is strict. Anything outside the contract returns None.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src import config, log
from src.llm import ainvoke_with_backoff
from src.config.prompts import RESPONSE_LABEL_SYSTEM
from src.feedback.reply_labeller import LabellingCallFailed, make_labeller_llm, strip_code_fence

logger = log.get_logger(__name__)

RESPONSE_LABELS: frozenset[str] = frozenset({"ANSWERED", "REFUSED", "NON_ANSWER", "UNCLEAR"})

RESPONSE_KEYS = frozenset({"label"})


def build_payload(*, question: str, response: str) -> str:
    """Render the question and the delivered reply as the JSON object the prompt describes.

    ensure_ascii is off so Russian text reaches the model as text rather than escapes.
    """
    return json.dumps({"question": question, "response": response}, ensure_ascii=False)


def parse_label(raw: str) -> str | None:
    """Parse the model's answer into one label, or None if it breaks the contract."""
    if not isinstance(raw, str):
        logger.warning("Response labelling: response content was not text")
        return None
    try:
        parsed = json.loads(strip_code_fence(raw))
    except (json.JSONDecodeError, TypeError):
        logger.warning("Response labelling: response was not JSON")
        return None
    if not isinstance(parsed, dict) or set(parsed.keys()) != RESPONSE_KEYS:
        logger.warning("Response labelling: response had the wrong shape")
        return None
    label = parsed["label"]
    return label if isinstance(label, str) and label in RESPONSE_LABELS else None


async def label_response(*, question: str, response: str) -> str | None:
    """Label one delivered reply against the question it answers.

    Returns:
        One of RESPONSE_LABELS, or None when the model answered outside the contract.

    Raises:
        LabellingCallFailed: The call never produced a response.
    """
    if not config.OPENROUTER_API_KEY:
        raise LabellingCallFailed("OPENROUTER_API_KEY unset")
    try:
        llm = make_labeller_llm(config.RESPONSE_LABEL_MAX_TOKENS)
        answer = await ainvoke_with_backoff(llm, [
            SystemMessage(content=RESPONSE_LABEL_SYSTEM),
            HumanMessage(content=build_payload(question=question, response=response)),
        ])
    except Exception as error:
        raise LabellingCallFailed(str(error) or type(error).__name__) from error
    return parse_label(answer.content or "")
