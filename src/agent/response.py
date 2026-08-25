"""ResponseAgent — personality LLM that turns worker facts into chat replies."""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src import config, log
from src.agent.middleware import (
    GroqContextGuard,
    ModelAttemptLogger,
    ThinkingStripper,
    guarded_ainvoke,
    should_retry,
)
from src.config.prompts import RESPONSE_PROMPT

logger = log.get_logger(__name__)


class ResponseAgent:
    """Manages the personality LLM that turns worker facts into chat replies.

    Mirrors WorkerAgent: owns a LangChain agent executor with retry/fallback
    middleware. Accepts an injectable ``response_executor`` for testing so
    production ``init()`` is never required in unit tests.
    """

    def __init__(self, *, response_executor=None) -> None:
        """Initialize with an optional pre-built executor.

        Args:
            response_executor: Pre-built agent executor (for testing).
        """
        self.__response_executor = response_executor

    async def init(self) -> None:
        """Build the response executor from configuration.

        Rebuilding resets middleware state so the slot returns to the primary model.
        """
        self.__response_executor = ResponseAgent.__build_executor()
        logger.info("ResponseAgent initialized with model: %s", config.RESPONSE_MODEL_FALLBACKS[0])

    async def invoke_response(self, messages: list, usage_sink: dict | None = None) -> str:
        """Run the response executor and return the final reply text.

        Think-block stripping is handled by ``ThinkingStripper`` middleware inside
        the executor. Language correction is handled upstream by
        ``LanguageCorrectionNode`` in the LangGraph pipeline.

        Args:
            messages: Message list (history + human turn). The executor prepends
                the system prompt internally; callers must not include it.
            usage_sink: If given, filled in-place with the reply message's
                ``usage_metadata`` (``input_tokens``/``output_tokens``/
                ``total_tokens``) when the serving model returned one —
                whichever model in the fallback chain actually answered.
                Left empty when the model reports none. Optional so existing
                callers and test doubles need no change.

        Returns:
            Reply text. Empty string when the model returns no content.

        Raises:
            RuntimeError: If called before ``init()``.
            ContextLengthError: If the prompt exceeds the model's context window.
            DailyLimitError: If all models have exhausted their daily token quota.
            RateLimitError: If rate-limit retries are exhausted on all models.
        """
        if self.__response_executor is None:
            raise RuntimeError("ResponseAgent.init() must be called before invoking response executor")
        result = await guarded_ainvoke(self.__response_executor, {"messages": messages})
        last_message = result["messages"][-1]
        if usage_sink is not None and last_message.usage_metadata:
            usage_sink.update(last_message.usage_metadata)
        return last_message.content or ""

    async def reset_model_index(self) -> None:
        """Rebuild the executor, resetting middleware state to the primary model."""
        await self.init()

    @staticmethod
    def __build_executor():
        """Build a response executor with retry/fallback middleware.

        The fallback chain always tries any remaining Groq models first, then
        falls over to RESPONSE_OPENROUTER_FALLBACKS on OpenRouter, in order —
        real cross-provider legs, so a Groq-wide outage degrades to a paid
        call instead of taking chat replies down entirely, and a single
        OpenRouter free model saturating doesn't either. Skipped when
        OPENROUTER_API_KEY is unset, same fail-open contract as
        filter_node.make_filter_llm.

        Returns:
            Configured LangChain agent executor.
        """
        fallback_llms = [
            ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0.7, max_tokens=1024, max_retries=0)
            for model in config.RESPONSE_MODEL_FALLBACKS[1:]
        ]
        if config.OPENROUTER_API_KEY:
            fallback_llms.extend(
                ChatOpenAI(
                    model=model,
                    api_key=config.OPENROUTER_API_KEY,
                    base_url=config.OPENROUTER_BASE_URL,
                    temperature=0.7,
                    max_tokens=1024,
                    max_retries=0,
                )
                for model in config.RESPONSE_OPENROUTER_FALLBACKS
            )
        else:
            logger.warning(
                "Response: OPENROUTER_API_KEY unset — no cross-provider fallback for %s",
                config.RESPONSE_MODEL_FALLBACKS[0],
            )
        primary_llm = ChatGroq(
            model=config.RESPONSE_MODEL_FALLBACKS[0],
            api_key=config.GROQ_API_KEY,
            temperature=0.7,
            max_tokens=1024,
            max_retries=0,
        )
        middleware = [
            ModelRetryMiddleware(retry_on=should_retry, on_failure="error", max_retries=3),
            ModelAttemptLogger(),
            GroqContextGuard(),
            ThinkingStripper(),
        ]
        if fallback_llms:
            middleware.insert(0, ModelFallbackMiddleware(*fallback_llms))
        executor = create_agent(
            primary_llm,
            [],
            system_prompt=RESPONSE_PROMPT,
            middleware=middleware,
        )
        logger.info("Response executor built with model: %s", config.RESPONSE_MODEL_FALLBACKS[0])
        return executor


response_agent = ResponseAgent()
