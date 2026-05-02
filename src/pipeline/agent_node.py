"""
AgentNode — fourth node in the LangGraph pipeline.

Builds an enriched prompt from AssembledContext (reply chain + user facts +
recent history) and invokes the main Agent executor to produce a reply.
"""

from langchain_core.callbacks import AsyncCallbackHandler

from src import log
from src.pipeline.state import BotState

logger = log.get_logger(__name__)

SEARCH_TOOLS = frozenset({"web_search", "fetch_article"})


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
        text = "🔗 Читаю страницу, подождите..." if tool_name == "fetch_article" else "🔍 Ищу, подождите немного..."
        try:
            await self.__message.reply_text(text)
        except Exception as err:
            logger.warning("Failed to send search notification: %s", err)

# How many recent-history messages to append when the reply chain is short.
RECENT_FILL_LIMIT = 10


class AgentNode:
    """Wraps the main Agent.run() with context-enriched input."""

    def __init__(self, agent) -> None:
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        context = state.get("context")
        user_input = msg["processed_text"] or msg["raw_text"] or ""
        username = msg["username"]
        chat_id = str(msg["chat_id"])

        enriched_input = self.__build_input(user_input, username, context)
        tg_message = msg["update"].message
        callback = SearchNotificationCallback(tg_message)

        try:
            response = await self.__agent.run(chat_id, username, enriched_input, callbacks=[callback])
        except Exception as err:
            logger.error("Agent execution failed in chat %s: %s", chat_id, err)
            raise

        return {"response": response}

    def __build_input(self, user_input: str, username: str, context) -> str:
        if not context:
            return f"{username}: {user_input}"

        parts: list[str] = []

        user_facts = context.get("user_facts") or {}
        if user_facts:
            parts.append("Что я знаю об участниках чата:")
            for uname, facts in user_facts.items():
                facts_line = "; ".join(facts)
                parts.append(f"@{uname}: {facts_line}")
            parts.append("")

        reply_chain = context.get("reply_chain") or []
        if reply_chain:
            parts.append("Цепочка ответов (от старого к новому):")
            for row in reply_chain:
                media_label = f" [{row['media_type']}]" if row["media_type"] != "text" else ""
                parts.append(f"@{row['username']}{media_label}: {row['content']}")
            parts.append("")
        else:
            recent = (context.get("recent_history") or [])[:RECENT_FILL_LIMIT]
            if recent:
                parts.append("Недавние сообщения чата:")
                for row in reversed(recent):
                    media_label = f" [{row['media_type']}]" if row["media_type"] != "text" else ""
                    parts.append(f"@{row['username']}{media_label}: {row['content']}")
                parts.append("")

        parts.append(f"@{username}: {user_input}")
        return "\n".join(parts)
