"""WorkerNode — agent that gathers facts using tools."""

import datetime
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage

from src import log
from src.agent import WORKER_MODEL_FALLBACKS, ContextLengthError, DailyLimitError, invoke_with_retry, strip_thinking
from src.pipeline.state import BotState
from src.store import unified_messages

logger = log.get_logger(__name__)

SEARCH_TOOLS = frozenset({"web_search"})
RECENT_FILL_LIMIT = 10


class SearchNotificationCallback(AsyncCallbackHandler):
    """Sends a Telegram notification before web_search or fetch_article runs.

    The sent Message object is appended to `holder` so the caller can later
    edit it with the final response instead of sending a second message.
    """

    def __init__(self, message, holder: list) -> None:
        super().__init__()
        self.__message = message
        self.__holder = holder
        self.__notified = False

    async def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "")
        if tool_name not in SEARCH_TOOLS or self.__notified:
            return
        self.__notified = True
        text = "🔍 Ищу, подожди немного..."
        try:
            sent = await self.__message.reply_text(text)
            self.__holder.append(sent)
        except Exception as err:
            logger.warning("Failed to send search notification: %s", err)


class WorkerNode:
    """Calls the worker agent and stores gathered facts in state."""

    def __init__(self, agent) -> None:
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        worker_input = self.__build_worker_input(msg, state.get("context"), state.get("response_trigger") or "explicit")
        notification_holder: list = []
        callback = SearchNotificationCallback(msg["update"].message, notification_holder)
        executor = self.__agent.get_worker_executor()
        run_config = {"callbacks": [callback]}

        for _ in range(len(WORKER_MODEL_FALLBACKS)):
            try:
                result = await invoke_with_retry(
                    executor,
                    {"messages": [HumanMessage(content=worker_input)]},
                    config=run_config,
                )
                output = strip_thinking(result["messages"][-1].content or "")
                notification_msg = notification_holder[0] if notification_holder else None
                return {"worker_output": output, "search_notification_msg": notification_msg}
            except DailyLimitError:
                if not await self.__agent.advance_worker_model():
                    raise
                executor = self.__agent.get_worker_executor()
            except ContextLengthError as err:
                logger.warning("Worker context too long: %s", err)
                return {"worker_output": "", "search_notification_msg": None}
            except Exception as err:
                logger.error("Worker failed: %s", err)
                return {"worker_output": "", "search_notification_msg": None}
        raise DailyLimitError("All fallback models exhausted in worker")

    def __build_worker_input(self, msg: dict, context, response_trigger: str = "explicit") -> str:
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
        content = row["content"]
        if row["media_type"] == "photo":
            content = unified_messages.display_photo_content(content)
        return f"@{row['username']}: {content}"
