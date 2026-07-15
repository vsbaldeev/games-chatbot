"""RoastAgent — dedicated LLM agent for roast text generation."""

import re

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent.language import (
    LANGUAGE_CORRECTION_PROMPT,
    needs_russian_correction,
    normalize_homoglyphs,
)
from src.agent.middleware import (
    GroqContextGuard,
    ThinkingStripper,
    guarded_ainvoke,
    should_retry,
    strip_thinking,
)
from src.config.prompts import ROAST_SYSTEM_PROMPT

logger = log.get_logger(__name__)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
MAX_ROAST_SENTENCES = 2
MIN_ROAST_PARAGRAPH_CHARS = 60


def trim_to_single_roast(text: str) -> str:
    """Reduce model output to one short roast.

    The model often ignores the "one roast, two sentences" instruction and stacks
    extra jokes — in further paragraphs ("Или вот ещё: ...") or as appended run-on
    sentences — and sometimes opens with a throwaway greeting paragraph before the
    real roast. This picks the first substantial paragraph (the roast, not a one-line
    greeting) and keeps only its first couple of sentences. Deterministic, because
    prompt instructions alone do not reliably stop either behaviour.

    Args:
        text: Raw roast text from the model.

    Returns:
        The first substantial paragraph capped at ``MAX_ROAST_SENTENCES`` sentences.
    """
    paragraphs = [block.strip() for block in text.strip().split("\n\n") if block.strip()]
    if not paragraphs:
        return ""
    roast = next(
        (block for block in paragraphs if len(block) >= MIN_ROAST_PARAGRAPH_CHARS),
        paragraphs[0],
    )
    sentences = SENTENCE_SPLIT_RE.split(roast)
    return " ".join(sentences[:MAX_ROAST_SENTENCES]).strip()


class RoastAgent:
    """LLM agent that generates roast text from assembled user facts.

    Mirrors ResponseAgent: owns a LangChain executor with retry/fallback middleware.
    Accepts an injectable ``roast_executor`` for testing so production ``init()``
    is never required in unit tests.
    """

    def __init__(self, *, roast_executor=None) -> None:
        """Initialize with an optional pre-built executor.

        Args:
            roast_executor: Pre-built agent executor (for testing).
        """
        self.__roast_executor = roast_executor

    async def init(self) -> None:
        """Build the roast executor from configuration."""
        self.__roast_executor = RoastAgent.__build_executor()
        logger.info("RoastAgent initialized with model: %s", config.ROAST_MODEL_FALLBACKS[0])

    async def invoke_roast(self, user_prompt: str) -> str:
        """Generate a roast from the assembled user prompt.

        Args:
            user_prompt: Formatted facts + username prompt string.

        Returns:
            Roast text in Russian. Empty string if the model returns no content.

        Raises:
            RuntimeError: If called before ``init()``.
            ContextLengthError: If the prompt exceeds the model's context window.
            DailyLimitError: If all models have exhausted their daily token quota.
            RateLimitError: If rate-limit retries are exhausted on all models.
        """
        if self.__roast_executor is None:
            raise RuntimeError("RoastAgent.init() must be called before invoking")
        reply = await self.__call_executor([HumanMessage(content=user_prompt)])
        corrected = await self.__apply_language_correction(user_prompt, reply)
        return trim_to_single_roast(corrected)

    async def reset_model_index(self) -> None:
        """Rebuild the executor, resetting middleware state to the primary model."""
        await self.init()

    async def __call_executor(self, messages: list) -> str:
        """Invoke the roast executor and return the last message content.

        Args:
            messages: LangChain message list to send to the executor.

        Returns:
            Text content of the last message, or empty string.
        """
        result = await guarded_ainvoke(self.__roast_executor, {"messages": messages})
        return result["messages"][-1].content or ""

    async def __apply_language_correction(self, user_prompt: str, reply: str) -> str:
        """Retry in Russian if the reply contains non-Russian script.

        Args:
            user_prompt: The original user prompt used for the first call.
            reply: The reply text to inspect for foreign script.

        Returns:
            Normalized reply, re-prompted in Russian when foreign script remains.
        """
        reply = normalize_homoglyphs(reply)
        visible = strip_thinking(reply)
        if not visible or not needs_russian_correction(visible):
            return reply
        logger.info("Foreign script detected in roast, retrying in Russian")
        correction_messages = [
            HumanMessage(content=user_prompt),
            AIMessage(content=reply),
            HumanMessage(content=LANGUAGE_CORRECTION_PROMPT),
        ]
        corrected = await self.__call_executor(correction_messages)
        return normalize_homoglyphs(corrected) if corrected else reply

    @staticmethod
    def __build_executor():
        """Build the roast executor with retry/fallback middleware.

        Returns:
            Configured LangChain agent executor.
        """
        # max_tokens is generous because the primary (gpt-oss-120b) is a reasoning
        # model: its hidden reasoning consumes output tokens before the answer, so a
        # tight cap leaves the visible content empty. trim_to_single_roast clamps the
        # final roast to two sentences regardless, so the headroom costs nothing visible.
        fallback_llms = [
            ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0.5, top_p=0.9, max_tokens=1024, max_retries=0)
            for model in config.ROAST_MODEL_FALLBACKS[1:]
        ]
        primary_llm = ChatGroq(
            model=config.ROAST_MODEL_FALLBACKS[0],
            api_key=config.GROQ_API_KEY,
            temperature=0.5,
            top_p=0.9,
            max_tokens=1024,
            max_retries=0,
        )
        return create_agent(
            primary_llm,
            [],
            system_prompt=ROAST_SYSTEM_PROMPT,
            middleware=[
                ModelFallbackMiddleware(*fallback_llms),
                ModelRetryMiddleware(retry_on=should_retry, on_failure="error", max_retries=3),
                GroqContextGuard(),
                ThinkingStripper(),
            ],
        )


roast_agent = RoastAgent()
