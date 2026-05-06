"""WorkerNode — domain-specific agent that gathers facts using tools."""

import datetime
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage

from src import log
from src.agent import AGENT_MODEL_FALLBACKS, DailyLimitError, invoke_with_retry
from src.pipeline.state import BotState

logger = log.get_logger(__name__)

SEARCH_TOOLS = frozenset({"web_search", "fetch_article"})
RECENT_FILL_LIMIT = 10


class SearchNotificationCallback(AsyncCallbackHandler):
    """Sends a Telegram notification before web_search or fetch_article runs."""

    def __init__(self, message) -> None:
        super().__init__()
        self.__message = message
        self.__notified = False

    async def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "")
        if tool_name not in SEARCH_TOOLS or self.__notified:
            return
        self.__notified = True
        if tool_name == "fetch_article":
            text = "🔗 Читаю страницу, подожди..."
        else:
            text = "🔍 Ищу, подожди немного..."
        try:
            await self.__message.reply_text(text)
        except Exception as err:
            logger.warning("Failed to send search notification: %s", err)


class WorkerNode:
    """Calls the domain-specific worker agent and stores gathered facts in state."""

    def __init__(self, agent, domain: str) -> None:
        self.__agent = agent
        self.__domain = domain

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        worker_input = self.__build_worker_input(msg, state.get("context"))
        callback = SearchNotificationCallback(msg["update"].message)
        executor = self.__agent.get_worker_executor(self.__domain)
        run_config = {"callbacks": [callback]}

        for _ in range(len(AGENT_MODEL_FALLBACKS)):
            try:
                result = await invoke_with_retry(
                    executor,
                    {"messages": [HumanMessage(content=worker_input)]},
                    config=run_config,
                )
                return {"worker_output": result["messages"][-1].content or ""}
            except DailyLimitError:
                if not await self.__agent.advance_model():
                    raise
                executor = self.__agent.get_worker_executor(self.__domain)
            except Exception as err:
                logger.error("Worker failed (domain=%s): %s", self.__domain, err)
                return {"worker_output": ""}
        raise DailyLimitError("All fallback models exhausted in worker")

    def __build_worker_input(self, msg: dict, context) -> str:
        user_input = msg["processed_text"] or msg["raw_text"] or ""
        username = msg["username"]
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        parts: list[str] = [f"Current datetime: {now}", ""]
        reply_chain = (context or {}).get("reply_chain") or []
        if reply_chain:
            parts.append("Context (reply chain):")
            for row in reply_chain:
                parts.append(f"@{row['username']}: {row['content']}")
            parts.append("")
        else:
            recent = ((context or {}).get("recent_history") or [])[:RECENT_FILL_LIMIT]
            if recent:
                parts.append("Recent chat context:")
                for row in reversed(recent):
                    parts.append(f"@{row['username']}: {row['content']}")
                parts.append("")
        parts.append(f"Question from @{username}: {user_input}")
        return "\n".join(parts)
