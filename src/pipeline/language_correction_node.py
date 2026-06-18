"""LanguageCorrectionNode — re-invokes the response agent when foreign script is detected."""

from langchain_core.messages import HumanMessage

from src import log
from src.agent import DailyLimitError, RateLimitError
from src.pipeline.state import BotState

logger = log.get_logger(__name__)

CORRECTION_PROMPT = (
    "Твой предыдущий ответ содержал символы не на русском языке. "
    "Ответь ТОЛЬКО на русском языке."
)


class LanguageCorrectionNode:
    """Re-invokes the response agent in Russian when foreign script slips through.

    Receives the original assembled message list from ``response_messages`` in
    state and appends a correction instruction before re-invoking.
    """

    def __init__(self, response_agent) -> None:
        """Initialize LanguageCorrectionNode.

        Args:
            response_agent: ``ResponseAgent`` instance used for the correction call.
        """
        self.__agent = response_agent

    async def __call__(self, state: BotState) -> dict:
        """Retry the response in Russian using the original message context.

        Args:
            state: Current pipeline state. ``response_messages`` must contain the
                assembled LangChain messages from the preceding response node.

        Returns:
            Dict with ``response`` key containing the corrected reply, or empty
            dict to keep the original response if correction itself fails.

        Raises:
            DailyLimitError: Propagated when the daily token quota is exhausted.
            RateLimitError: Propagated when rate-limit retries are exhausted.
        """
        messages = state.get("response_messages") or []
        correction_messages = messages + [HumanMessage(content=CORRECTION_PROMPT)]
        try:
            corrected = await self.__agent.invoke_response(correction_messages)
            return {"response": corrected}
        except (DailyLimitError, RateLimitError):
            raise
        except Exception as err:
            logger.warning("Language correction failed: %s", err)
            return {}
