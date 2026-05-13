"""IntentClassifierNode — classifies message intent to route to the right worker."""

from langchain_core.messages import HumanMessage, SystemMessage

from src import log
from src.pipeline.state import BotState

logger = log.get_logger(__name__)

CLASSIFY_SYSTEM = (
    "Classify the user message into ONE word: games, media, or general.\n"
    "games = video games, consoles, PS5, Xbox, Steam, PC gaming, game recommendations\n"
    "media = movies, TV shows, series, anime, cartoons, manga, streaming\n"
    "general = everything else (chat, dates, bot commands, greetings, etc.)\n"
    "Reply with ONLY the single classification word, nothing else."
)

VALID_INTENTS = frozenset({"games", "media", "general"})


class IntentClassifierNode:
    def __init__(self, agent) -> None:
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        text = state["incoming"]["processed_text"] or state["incoming"]["raw_text"] or ""
        intent = await self.__classify(text, state["incoming"]["message_id"])
        return {"intent": intent}

    async def __classify(self, text: str, message_id: int) -> str:
        llm = self.__agent.get_classifier_llm()
        try:
            response = await llm.ainvoke([
                SystemMessage(content=CLASSIFY_SYSTEM),
                HumanMessage(content=text),
            ])
            raw = (response.content if isinstance(response.content, str) else "").strip().lower()
            if raw in VALID_INTENTS:
                logger.info("Intent '%s' for message %s (model: %s)", raw, message_id, llm.model_name)
                return raw
            logger.warning(
                "Unexpected classifier output %r for message %s — defaulting to general",
                raw, message_id,
            )
            return "general"
        except Exception as err:
            logger.warning("Intent classification failed for message %s, defaulting to general: %s", message_id, err)
            return "general"
