"""WorkerAgent — tool-calling executor that gathers facts for the pipeline."""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import HumanMessage
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

logger = log.get_logger(__name__)

WORKER_MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",                        # primary:    120B, best tool-call quality
    "qwen/qwen3-32b",                             # fallback-1: 32B,  parallel tools ✅
    "openai/gpt-oss-20b",                         # fallback-2: 20B,  structured tool caller
    "meta-llama/llama-4-scout-17b-16e-instruct",  # fallback-3: 17B,  last resort, parallel tools ✅
]

WORKER_PROMPT = """You are a data-gathering assistant. Call tools to fetch facts when needed.
Output findings as plain text. No personality, no sarcasm.
Use English for tool queries; output language should match the user's question.

CONTEXT FIRST: If the reply chain already contains the content the user is asking about
(a forwarded post, article text, or message), extract the key facts from it directly.
Do NOT call any tools for content already present in the context.

TOOL SELECTION:
- Game details, platforms, genres, rating, developer: search_games → get_game_details
- Steam player count: get_steam_player_count
- Steam price and store details: get_steam_app_details
- Critic and user review scores: get_game_reviews, get_steam_reviews_summary
- Top PS5 game recommendations: get_ps5_recommendations
- PS Store price in Turkish lira: get_ps_store_price_tr, get_ps_store_sales
- Movie or animated film: search_movie_or_tv (type "movie")
- TV series or animated series: search_movie_or_tv (type "tv")
- Anime details, episodes, score, studios: search_anime
- Any other factual question, news, release dates, crossplay: web_search

If no tools are needed and no facts to extract (casual chat, reactions, bot commands, greetings), output an empty string.

STRICT: call ALL needed tools BEFORE writing any text. NEVER output text between tool calls.
Output raw facts only — no conversational wrapping."""


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
        logger.info("WorkerAgent initialized with model: %s", WORKER_MODEL_FALLBACKS[0])

    async def invoke_worker(self, prompt: str, *, callbacks=None) -> str:
        """Run the executor and return the output text.

        Think-block stripping is handled by ``ThinkingStripper`` middleware inside
        the executor, so the returned string is already clean.

        Args:
            prompt: Assembled worker-input string to send as the human message.
            callbacks: Optional list of LangChain callbacks (e.g. for notifications).

        Returns:
            Worker output with ``<think>`` blocks already removed by middleware.
            Empty string when the model returns no content.

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
        return result["messages"][-1].content or ""

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
            ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0.3, max_tokens=1024)
            for model in WORKER_MODEL_FALLBACKS[1:]
        ]
        primary_llm = ChatGroq(
            model=WORKER_MODEL_FALLBACKS[0],
            api_key=config.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=1024,
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
        logger.info("Worker executor built with model: %s", WORKER_MODEL_FALLBACKS[0])
        return executor


worker_agent = WorkerAgent()
