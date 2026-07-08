"""WorkerAgent — tool-calling executor that gathers facts for the pipeline."""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent.exceptions import ContextLengthError, DailyLimitError, RateLimitError
from src.agent.middleware import (
    GroqContextGuard,
    ThinkingStripper,
    ToolMessageSanitizer,
    guarded_ainvoke,
    should_retry,
)
from src.config.prompts import WORKER_PROMPT

logger = log.get_logger(__name__)


class WorkerAgent:
    """Manages the tool-calling executor that gathers facts for the pipeline.

    Owns the LangChain agent with retry/fallback middleware.  Accepts an
    injectable ``worker_executor`` for testing so production ``init()`` is never
    required in unit tests.
    """

    def __init__(self, *, worker_executor=None) -> None:
        """Initialize with an optional pre-built executor.

        Args:
            worker_executor: Pre-built agent executor (for testing).
        """
        self.__worker_executor = worker_executor

    async def init(self) -> None:
        """Build the worker executor from configuration.

        Rebuilding resets middleware state so the slot returns to the primary model.
        """
        self.__worker_executor = WorkerAgent.__build_executor()
        logger.info("WorkerAgent initialized with model: %s", config.WORKER_MODEL_FALLBACKS[0])

    async def invoke_worker(self, prompt: str, *, callbacks=None) -> tuple[str, bool]:
        """Run the executor and return the output text with tool provenance.

        Think-block stripping is handled by ``ThinkingStripper`` middleware inside
        the executor, so the returned string is already clean. Whether any tool
        actually ran is determined mechanically by scanning the result messages
        for ``ToolMessage`` instances — not by model judgment.

        Args:
            prompt: Assembled worker-input string to send as the human message.
            callbacks: Optional list of LangChain callbacks (e.g. for notifications).

        Returns:
            Tuple of ``(output, tools_used)``: the worker output with
            ``<think>`` blocks already removed (empty string when the model
            returned no content), and True when at least one tool call ran.

        Raises:
            RuntimeError: If called before ``init()``.
            ContextLengthError: If the prompt exceeds the model's context window.
            DailyLimitError: If all models have exhausted their daily token quota.
            RateLimitError: If rate-limit retries are exhausted on all models.
        """
        if self.__worker_executor is None:
            raise RuntimeError("WorkerAgent.init() must be called before invoking worker")
        run_config = {"callbacks": callbacks} if callbacks else {}
        result = await guarded_ainvoke(
            self.__worker_executor,
            {"messages": [HumanMessage(content=prompt)]},
            config=run_config,
        )
        messages = result["messages"]
        tools_used = any(isinstance(message, ToolMessage) for message in messages)
        return messages[-1].content or "", tools_used

    async def reset_model_index(self) -> None:
        """Rebuild the executor, resetting middleware state to the primary model."""
        await self.init()

    @staticmethod
    def __build_executor():
        """Build a worker executor with retry/fallback middleware.

        Returns:
            Configured LangChain agent executor.
        """
        from src.tools import ALL_TOOLS
        fallback_llms = [
            ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0.3, max_tokens=1024, max_retries=0)
            for model in config.WORKER_MODEL_FALLBACKS[1:]
        ]
        primary_llm = ChatGroq(
            model=config.WORKER_MODEL_FALLBACKS[0],
            api_key=config.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=1024,
            max_retries=0,
        )
        executor = create_agent(
            primary_llm,
            ALL_TOOLS,
            system_prompt=WORKER_PROMPT,
            middleware=[
                ModelFallbackMiddleware(*fallback_llms),
                ModelRetryMiddleware(retry_on=should_retry, on_failure="error", max_retries=3),
                GroqContextGuard(),
                ToolMessageSanitizer(),
                ThinkingStripper(),
            ],
        )
        logger.info("Worker executor built with model: %s", config.WORKER_MODEL_FALLBACKS[0])
        return executor


worker_agent = WorkerAgent()
