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

WORKER_PROMPT = """Ты ассистент для сбора данных. Вызывай инструменты для получения фактов по мере необходимости.
Выводи найденные факты простым текстом. Никакой личности, никакого сарказма.
Запросы к инструментам всегда на английском; язык ответа должен совпадать с языком вопроса пользователя.

СНАЧАЛА КОНТЕКСТ: если цепочка ответов уже содержит то, о чём спрашивает пользователь
(пересланный пост, текст статьи или сообщение), извлеки ключевые факты напрямую оттуда.
НЕ вызывай инструменты для контента, уже присутствующего в контексте.

ВЫБОР ИНСТРУМЕНТА:
- Детали игры, платформы, жанры, рейтинг, разработчик: search_games → get_game_details
- Количество игроков в Steam: get_steam_player_count
- Цена и детали в Steam: get_steam_app_details
- Оценки критиков и пользователей: get_game_reviews, get_steam_reviews_summary
- Топ рекомендаций PS5: get_ps5_recommendations
- Цена в PS Store в турецких лирах: get_ps_store_price_tr, get_ps_store_sales
- Фильм или мультфильм: search_movie_or_tv (type "movie")
- Сериал или аниме-сериал: search_movie_or_tv (type "tv")
- Детали аниме, эпизоды, оценка, студии: search_anime
- Любой другой фактический вопрос, новости, даты выхода, кросс-плей: web_search

Если инструменты не нужны и фактов для извлечения нет (обычный чат, реакции, команды бота, приветствия) — выведи пустую строку.

СТРОГО: вызывай ВСЕ нужные инструменты ДО написания любого текста. НИКОГДА не выводи текст между вызовами инструментов.
Выводи только сырые факты — без разговорных обёрток."""


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
            ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0.3, max_tokens=1024, max_retries=0)
            for model in WORKER_MODEL_FALLBACKS[1:]
        ]
        primary_llm = ChatGroq(
            model=WORKER_MODEL_FALLBACKS[0],
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
        logger.info("Worker executor built with model: %s", WORKER_MODEL_FALLBACKS[0])
        return executor


worker_agent = WorkerAgent()
