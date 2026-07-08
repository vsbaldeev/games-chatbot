"""ComedianAgent — autonomous-humor decision brain.

Given the live conversation and what is known about the participants, the
comedian decides whether to drop a joke that *spawns* conversation, picks the
register (light vs roast), or stays silent. It returns a strict JSON decision;
parsing is fail-safe to silence, so a malformed, empty, or non-Russian answer
becomes an abstain rather than a bad message in the chat.
"""

import json
import re
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent.language import needs_russian_correction, normalize_homoglyphs
from src.agent.middleware import (
    GroqContextGuard,
    ThinkingStripper,
    guarded_ainvoke,
    should_retry,
    strip_thinking,
)
from src.agent.roast import trim_to_single_roast
from src.config.prompts import COMEDIAN_SYSTEM_PROMPT

logger = log.get_logger(__name__)

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
VALID_REGISTERS = ("light", "roast")

@dataclass
class ComedianDecision:
    """The comedian's decision for one moment.

    Attributes:
        act: True when a joke should be sent.
        register: ``"light"`` or ``"roast"``.
        text: The joke text (empty when abstaining).
        reply_to_message_id: Id of the message the joke targets (cited from the
            ``[#id]`` markers in the conversation), or None when the joke is
            about the conversation as a whole. Not yet validated against the
            actual message set — the humor node does that.
    """

    act: bool
    register: str
    text: str
    reply_to_message_id: int | None = None

    @classmethod
    def abstain(cls) -> "ComedianDecision":
        """Return a do-nothing decision."""
        return cls(act=False, register="light", text="")


def parse_reply_target(value: object) -> int | None:
    """Coerce the model-provided ``reply_to`` value into a message id.

    Accepts an int or a numeric string (with an optional ``#`` prefix, since
    models sometimes echo the marker format). Anything else — null, booleans,
    prose — becomes None, which the caller treats as "no anchor".

    Args:
        value: Raw ``reply_to`` value from the decision JSON.

    Returns:
        The message id, or None when the value is unusable.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = value.strip().lstrip("#")
        if digits.isdigit():
            return int(digits)
    return None


def parse_decision(raw: str) -> ComedianDecision:
    """Parse the model's raw output into a decision, failing safe to silence.

    Args:
        raw: Raw model output (may contain a thinking block or surrounding prose).

    Returns:
        A valid acting decision, or ``ComedianDecision.abstain()`` for any output
        that is malformed, not acting, empty, or not in Russian.
    """
    visible = strip_thinking(raw or "")
    match = JSON_OBJECT_RE.search(visible)
    if not match:
        return ComedianDecision.abstain()
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return ComedianDecision.abstain()
    if not isinstance(data, dict) or data.get("act") is not True:
        return ComedianDecision.abstain()
    text = normalize_homoglyphs(trim_to_single_roast(str(data.get("text") or "")))
    if not text or needs_russian_correction(text):
        return ComedianDecision.abstain()
    register = data.get("register")
    if register not in VALID_REGISTERS:
        register = "light"
    reply_to = parse_reply_target(data.get("reply_to"))
    return ComedianDecision(act=True, register=register, text=text, reply_to_message_id=reply_to)


def build_comedian_prompt(conversation: str, material: str) -> str:
    """Assemble the user turn for the comedian.

    Args:
        conversation: Recent chat messages, oldest-first, already rendered.
        material: Formatted participant material (facts/quotes/role/stats), or "".

    Returns:
        The prompt string to send as the human turn.
    """
    parts = ["Разговор в чате сейчас:", conversation or "(пусто)", ""]
    if material.strip():
        parts += ["Что известно об участниках:", material, ""]
    parts.append(
        "Реши: стоит ли вкинуть шутку, которая заведёт движ? "
        "Ответь строго одним JSON-объектом и больше ничем."
    )
    return "\n".join(parts)


class ComedianAgent:
    """LLM agent that decides whether and how to inject humor.

    Mirrors ``RoastAgent``: owns a LangChain executor with retry/fallback
    middleware and accepts an injectable executor for testing.
    """

    def __init__(self, *, comedian_executor=None) -> None:
        """Initialize with an optional pre-built executor.

        Args:
            comedian_executor: Pre-built agent executor (for testing).
        """
        self.__executor = comedian_executor

    async def init(self) -> None:
        """Build the comedian executor from configuration."""
        self.__executor = ComedianAgent.__build_executor()
        logger.info("ComedianAgent initialized with model: %s", config.COMEDIAN_MODEL_FALLBACKS[0])

    async def decide(self, conversation: str, material: str) -> ComedianDecision:
        """Decide whether to joke about the current moment.

        Args:
            conversation: Recent chat messages, already rendered.
            material: Formatted participant material, or "".

        Returns:
            The parsed decision (abstain on any unusable output).

        Raises:
            RuntimeError: If called before ``init()``.
            ContextLengthError: If the prompt exceeds the model's context window.
            DailyLimitError: If all models have exhausted their daily token quota.
            RateLimitError: If rate-limit retries are exhausted on all models.
        """
        if self.__executor is None:
            raise RuntimeError("ComedianAgent.init() must be called before deciding")
        prompt = build_comedian_prompt(conversation, material)
        result = await guarded_ainvoke(self.__executor, {"messages": [HumanMessage(content=prompt)]})
        raw = result["messages"][-1].content or ""
        return parse_decision(raw)

    async def reset_model_index(self) -> None:
        """Rebuild the executor, resetting middleware state to the primary model."""
        await self.init()

    @staticmethod
    def __build_executor():
        """Build the comedian executor with retry/fallback middleware.

        Returns:
            Configured LangChain agent executor.
        """
        fallback_llms = [
            ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0.8, top_p=0.95, max_tokens=1024, max_retries=0)
            for model in config.COMEDIAN_MODEL_FALLBACKS[1:]
        ]
        primary_llm = ChatGroq(
            model=config.COMEDIAN_MODEL_FALLBACKS[0],
            api_key=config.GROQ_API_KEY,
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            max_retries=0,
        )
        return create_agent(
            primary_llm,
            [],
            system_prompt=COMEDIAN_SYSTEM_PROMPT,
            middleware=[
                ModelFallbackMiddleware(*fallback_llms),
                ModelRetryMiddleware(retry_on=should_retry, on_failure="error", max_retries=3),
                GroqContextGuard(),
                ThinkingStripper(),
            ],
        )


comedian_agent = ComedianAgent()
