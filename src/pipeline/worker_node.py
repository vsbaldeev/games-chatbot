"""WorkerNode — agent that gathers facts using tools."""

import datetime
from typing import Literal

from langchain_core.callbacks import AsyncCallbackHandler

from src import log
from src.agent import ContextLengthError, DailyLimitError, RateLimitError
from src.pipeline.response_node import row_speaker
from src.pipeline.state import BotState
from src.store import unified_messages

logger = log.get_logger(__name__)

SEARCH_TOOLS = frozenset({"web_search"})
RECENT_FILL_LIMIT = 10


class SearchNotificationCallback(AsyncCallbackHandler):
    """Sends a Telegram notification before web_search runs.

    The sent Message object is stored as ``sent_message`` so the caller can
    later edit it with the final response instead of sending a second message.
    """

    def __init__(self, message) -> None:
        """Initialize the callback.

        Args:
            message: Telegram Message object used to send the notification reply.
        """
        super().__init__()
        self.__message = message
        self.__notified = False
        self.sent_message = None

    async def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Send a search notification on the first web_search tool invocation.

        Args:
            serialized: Tool metadata dict containing at least a ``name`` key.
            input_str: Raw input string passed to the tool (unused).
            **kwargs: Additional keyword arguments forwarded by LangChain.
        """
        tool_name = serialized.get("name", "")
        if tool_name not in SEARCH_TOOLS or self.__notified:
            return
        self.__notified = True
        try:
            self.sent_message = await self.__message.reply_text("🔍 Ищу, подожди немного...")
        except Exception as err:
            logger.warning("Failed to send search notification: %s", err)


class WorkerNode:
    """Calls the worker agent and stores gathered facts in state."""

    def __init__(self, agent) -> None:
        """Initialize the worker node.

        Args:
            agent: Agent instance providing the worker executor and fallback logic.
        """
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        """Run the worker agent and return gathered facts.

        Absorbs ContextLengthError and unexpected exceptions by returning empty
        output so the pipeline can still attempt a response. DailyLimitError and
        RateLimitError are re-raised for the top-level handler to surface to the user.

        Args:
            state: Current pipeline state containing the incoming message and context.

        Returns:
            Dict with keys ``worker_output`` (str), ``search_notification_msg``
            (the sent Telegram Message or None) and ``worker_tools_used``
            (True when at least one tool actually ran; False on all error
            and skip paths).

        Raises:
            DailyLimitError: Propagated from ``agent.invoke_worker`` on daily quota exhaustion.
            RateLimitError: Propagated from ``agent.invoke_worker`` on transient rate limits.
        """
        if (
            state.get("is_bot_insult")
            or state.get("wind_down")
            or state.get("photo_request")
            or state.get("response_trigger") == "youtube_short"
        ):
            # A comeback needs personality, not IGDB lookups: skipping the
            # worker avoids a «🔍 Ищу…» notification before the burn and junk
            # search output steering it. Same for Shorts summaries — the
            # source material is already in processed_text — for wind-down
            # brush-offs, which are one short phrase closing the conversation,
            # and for photo-request acks («ща сфоткаю»), which must go out
            # fast and without a search notification.
            return {"worker_output": "", "search_notification_msg": None, "worker_tools_used": False}
        msg = state["incoming"]
        worker_input = self.__build_worker_input(msg, state.get("context"), state.get("response_trigger") or "explicit")
        tg_message = msg["update"].message
        callback = SearchNotificationCallback(tg_message) if tg_message is not None else None
        callbacks = [callback] if callback is not None else None

        try:
            output, tools_used = await self.__agent.invoke_worker(worker_input, callbacks=callbacks)
            notification_msg = callback.sent_message if callback is not None else None
            return {
                "worker_output": output,
                "search_notification_msg": notification_msg,
                "worker_tools_used": tools_used,
            }
        except ContextLengthError as err:
            logger.warning("Worker context too long: %s", err)
            return {"worker_output": "", "search_notification_msg": None, "worker_tools_used": False}
        except (DailyLimitError, RateLimitError):
            raise
        except Exception as err:
            logger.error("Worker failed: %s", err, exc_info=True)
            return {"worker_output": "", "search_notification_msg": None, "worker_tools_used": False}

    def __build_worker_input(self, msg: dict, context, response_trigger: Literal["explicit", "insult_check", "random"] = "explicit") -> str:
        """Assemble the prompt string sent to the worker agent.

        Includes the current UTC datetime, reply chain or recent history as
        context (mutually exclusive), and the user's question. Recent history
        is omitted for random triggers to avoid injecting unrelated chat noise.

        Args:
            msg: IncomingMessage dict for the current update.
            context: Context dict with ``reply_chain`` and ``recent_history`` lists,
                or None if context is unavailable.
            response_trigger: ``"explicit"`` when the bot was @mentioned or replied
                to; ``"insult_check"`` for confirmed bot-word insults (treated
                like explicit); ``"random"`` for unprompted media responses.

        Returns:
            Assembled prompt string ready to send to the worker executor.
        """
        user_input = msg["processed_text"] or msg["raw_text"] or ""
        username = msg["username"]
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        parts: list[str] = [f"Current datetime: {now}", ""]
        reply_chain = (context or {}).get("reply_chain") or []
        if reply_chain:
            parts.append("Context (reply chain):")
            for row in reply_chain:
                parts.append(self.__render_row(row))
            parts.append("")
        elif response_trigger != "random":
            recent = ((context or {}).get("recent_history") or [])[:RECENT_FILL_LIMIT]
            if recent:
                parts.append("Recent chat context:")
                for row in reversed(recent):
                    parts.append(self.__render_row(row))
                parts.append("")
        parts.append(f"Question from @{username}: {user_input}")
        return "\n".join(parts)

    @staticmethod
    def __render_row(row: dict) -> str:
        """Format a message row as ``speaker: content`` for the worker prompt.

        The bot's own past messages are labelled ``Ты (бот)`` rather than
        ``@username`` so the worker never treats them as another participant.

        Args:
            row: Message row dict with ``user_id``, ``username``, ``content``,
                and ``media_type`` keys.

        Returns:
            Formatted string representation of the message.
        """
        content = unified_messages.display_media_content(row["media_type"], row["content"])
        return f"{row_speaker(row)}: {content}"
