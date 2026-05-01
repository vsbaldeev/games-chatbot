"""
AgentNode — fourth node in the LangGraph pipeline.

Builds an enriched prompt from AssembledContext (reply chain + user facts +
recent history) and invokes the main Agent executor to produce a reply.
"""

from src import log

from src.pipeline.state import BotState

logger = log.get_logger(__name__)

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

        try:
            response = await self.__agent.run(chat_id, username, enriched_input)
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
