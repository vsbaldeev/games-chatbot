"""RoastAgent — dedicated LLM agent for roast text generation."""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent.language import FOREIGN_SCRIPT_RE, LANGUAGE_CORRECTION_PROMPT
from src.agent.middleware import (
    GroqContextGuard,
    ThinkingStripper,
    guarded_ainvoke,
    should_retry,
    strip_thinking,
)

logger = log.get_logger(__name__)

ROAST_MODEL_FALLBACKS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]

ROAST_SYSTEM_PROMPT = (
    "Ты — стендап-комик, который жарит зрителя перед пёстрым залом. "
    "Тебе дают факты о человеке — уже отобранные, самые неловкие. "
    "В зале люди разного возраста и круга — шутка должна зайти любому, "
    "без знания игр, аниме, мемов, сериалов или музыки. "
    "Выбери самый понятный и нелепый факт "
    "и обыграй его простыми словами, как бытовую сценку из жизни. "
    "Не делай отсылок к фэндомам, не цитируй названия песен, игр или персонажей — "
    "если шутку поймёт только фанат, она провалилась. "
    "Коротко задай контекст, потом одна неожиданная фраза точно в цель. "
    "Не объясняй шутку. Только русский. Мат допустим."
)


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
        logger.info("RoastAgent initialized with model: %s", ROAST_MODEL_FALLBACKS[0])

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
        return await self.__apply_language_correction(user_prompt, reply)

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
            Corrected reply if foreign script was detected, otherwise the original.
        """
        visible = strip_thinking(reply)
        if not visible or not FOREIGN_SCRIPT_RE.search(visible):
            return reply
        logger.warning("Foreign script detected in roast, retrying in Russian")
        correction_messages = [
            HumanMessage(content=user_prompt),
            AIMessage(content=reply),
            HumanMessage(content=LANGUAGE_CORRECTION_PROMPT),
        ]
        return await self.__call_executor(correction_messages) or reply

    @staticmethod
    def __build_executor():
        """Build the roast executor with retry/fallback middleware.

        Returns:
            Configured LangChain agent executor.
        """
        fallback_llms = [
            ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0.5, top_p=0.9, max_tokens=100, max_retries=0)
            for model in ROAST_MODEL_FALLBACKS[1:]
        ]
        primary_llm = ChatGroq(
            model=ROAST_MODEL_FALLBACKS[0],
            api_key=config.GROQ_API_KEY,
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
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
