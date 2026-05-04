"""IntentClassifierNode — classifies message intent to route to the right worker."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import config, log
from src.pipeline.state import BotState

logger = log.get_logger(__name__)

CLASSIFY_SYSTEM = (
    "Classify the user message into ONE word: games, media, or general.\n"
    "games = video games, consoles, PS5, Xbox, Steam, PC gaming, game recommendations\n"
    "media = movies, TV shows, series, anime, cartoons, manga, streaming\n"
    "general = everything else (chat, dates, bot commands, greetings, etc.)\n"
    "Reply with ONLY the single classification word, nothing else."
)


class IntentClassifierNode:
    def __init__(self) -> None:
        self.__llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=config.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=5,
        )

    async def __call__(self, state: BotState) -> dict:
        text = state["incoming"]["processed_text"] or state["incoming"]["raw_text"] or ""
        intent = await self.__classify(text)
        logger.info("Intent: '%s' for: %.80s", intent, text)
        return {"intent": intent}

    async def __classify(self, text: str) -> str:
        try:
            response = await self.__llm.ainvoke([
                SystemMessage(content=CLASSIFY_SYSTEM),
                HumanMessage(content=text),
            ])
            result = response.content.strip().lower()
            if result in ("games", "media"):
                return result
            return "general"
        except Exception as err:
            logger.warning("Intent classification failed, defaulting to general: %s", err)
            return "general"
