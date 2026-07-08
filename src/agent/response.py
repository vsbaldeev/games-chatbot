"""ResponseAgent — personality LLM that turns worker facts into chat replies."""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_groq import ChatGroq

from src import config, log
from src.agent.middleware import (
    GroqContextGuard,
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

    async def invoke_response(self, messages: list) -> str:
        """Run the response executor and return the final reply text.

        Think-block stripping is handled by ``ThinkingStripper`` middleware inside
        the executor. Language correction is handled upstream by
        ``LanguageCorrectionNode`` in the LangGraph pipeline.

        Args:
            messages: Message list (history + human turn). The executor prepends
                the system prompt internally; callers must not include it.

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
        return result["messages"][-1].content or ""

    async def reset_model_index(self) -> None:
        """Rebuild the executor, resetting middleware state to the primary model."""
        await self.init()

    @staticmethod
    def __build_executor():
        """Build a response executor with retry/fallback middleware.

        Returns:
            Configured LangChain agent executor.
        """
        fallback_llms = [
            ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0.7, max_tokens=1024, max_retries=0)
            for model in config.RESPONSE_MODEL_FALLBACKS[1:]
        ]
        primary_llm = ChatGroq(
            model=config.RESPONSE_MODEL_FALLBACKS[0],
            api_key=config.GROQ_API_KEY,
            temperature=0.7,
            max_tokens=1024,
            max_retries=0,
        )
        executor = create_agent(
            primary_llm,
            [],
            system_prompt=RESPONSE_PROMPT,
            middleware=[
                ModelFallbackMiddleware(*fallback_llms),
                ModelRetryMiddleware(retry_on=should_retry, on_failure="error", max_retries=3),
                GroqContextGuard(),
                ThinkingStripper(),
            ],
        )
        logger.info("Response executor built with model: %s", config.RESPONSE_MODEL_FALLBACKS[0])
        return executor


response_agent = ResponseAgent()
