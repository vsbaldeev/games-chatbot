"""ResponseNode — personality LLM that turns worker facts into a chat reply."""

import asyncio
import datetime

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage

from src import config, log
from src.agent import AGENT_MODEL_FALLBACKS, DailyLimitError, RESPONSE_PROMPT, apply_language_correction
from src.pipeline.state import BotState

logger = log.get_logger(__name__)

RECENT_FILL_LIMIT = 10


def build_history(chat_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=chat_id,
        connection=config.SQLALCHEMY_DB_URL,
        table_name="message_store",
    )


async def trim_history(history: SQLChatMessageHistory, max_messages: int) -> None:
    def trim_sync() -> None:
        messages = history.messages
        if len(messages) <= max_messages:
            return
        to_keep = messages[-max_messages:]
        history.clear()
        for msg in to_keep:
            history.add_message(msg)
    await asyncio.to_thread(trim_sync)


async def trim_db_history(history: SQLChatMessageHistory, max_user_messages: int = 40) -> None:
    def trim_sync() -> None:
        messages = history.messages
        user_indices = [idx for idx, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
        if len(user_indices) <= max_user_messages:
            return
        cutoff = user_indices[-max_user_messages]
        to_keep = messages[cutoff:]
        history.clear()
        for msg in to_keep:
            history.add_message(msg)
    await asyncio.to_thread(trim_sync)


def build_response_input(username: str, user_input: str, worker_output: str, context) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [f"Текущая дата и время: {now}", ""]
    user_facts = (context or {}).get("user_facts") or {}
    if user_facts:
        parts.append("Что я знаю об участниках чата:")
        for uname, facts in user_facts.items():
            parts.append(f"@{uname}: {'; '.join(facts)}")
        parts.append("")
    reply_chain = (context or {}).get("reply_chain") or []
    if reply_chain:
        parts.append("Цепочка ответов (от старого к новому):")
        for row in reply_chain:
            media_label = f" [{row['media_type']}]" if row["media_type"] != "text" else ""
            parts.append(f"@{row['username']}{media_label}: {row['content']}")
        parts.append("")
    else:
        recent = ((context or {}).get("recent_history") or [])[:RECENT_FILL_LIMIT]
        if recent:
            parts.append("Недавние сообщения чата:")
            for row in reversed(recent):
                media_label = f" [{row['media_type']}]" if row["media_type"] != "text" else ""
                parts.append(f"@{row['username']}{media_label}: {row['content']}")
            parts.append("")
    if worker_output:
        parts.append(f"[Собранные данные]:\n{worker_output}\n")
    parts.append(f"@{username}: {user_input}")
    return "\n".join(parts)


class ResponseNode:
    """Generates the final personality-driven reply from gathered worker facts."""

    def __init__(self, agent) -> None:
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        history = build_history(str(msg["chat_id"]))
        await trim_history(history, config.MAX_HISTORY_MESSAGES)
        past_messages = await asyncio.to_thread(lambda: history.messages)
        enriched = build_response_input(
            msg["username"],
            msg["processed_text"] or msg["raw_text"] or "",
            state.get("worker_output") or "",
            state.get("context"),
        )
        ai_message = await self.__generate(past_messages, enriched)
        await self.__save(history, enriched, ai_message)
        return {"response": ai_message.content if ai_message else ""}

    async def __generate(self, past_messages: list, enriched: str):
        messages = [SystemMessage(content=RESPONSE_PROMPT)] + past_messages + [HumanMessage(content=enriched)]
        llm = self.__agent.get_response_llm()
        for _ in range(len(AGENT_MODEL_FALLBACKS)):
            try:
                ai_message = await llm.ainvoke(messages)
                return await apply_language_correction(llm, ai_message, messages)
            except Exception as err:
                error_str = str(err).lower()
                is_daily = any(phrase in error_str for phrase in ("per day", "daily", "tokens_per_day"))
                if is_daily and await self.__agent.advance_model():
                    llm = self.__agent.get_response_llm()
                    continue
                raise
        raise DailyLimitError("All fallback models exhausted in response node")

    async def __save(self, history, enriched: str, ai_message) -> None:
        if not (ai_message and ai_message.content and ai_message.content.strip()):
            return
        def save_sync() -> None:
            history.add_user_message(enriched)
            history.add_message(ai_message)
        await asyncio.to_thread(save_sync)
        await trim_db_history(history)
