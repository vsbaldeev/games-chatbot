"""ResponseNode — personality LLM that turns worker facts into a chat reply."""

import datetime
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src import config, log
from src.agent import AGENT_MODEL_FALLBACKS, DailyLimitError, RESPONSE_PROMPT, apply_language_correction, strip_thinking
from src.pipeline.state import BotState
from src.store import thread_history, unified_messages

logger = log.get_logger(__name__)

RECENT_FILL_LIMIT = 10
TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text, flags=re.DOTALL)
    lines = [line for line in text.splitlines() if not TABLE_SEP_RE.match(line)]
    return "\n".join(lines)


def _render_row(row: dict) -> str:
    media_type = row["media_type"]
    content = row["content"]
    if media_type == "photo":
        content = unified_messages.display_photo_content(content)
    media_label = f" [{media_type}]" if media_type != "text" else ""
    return f"@{row['username']}{media_label}: {content}"


def _build_past_messages(history: list[dict]) -> list[HumanMessage | AIMessage]:
    result: list[HumanMessage | AIMessage] = []
    for entry in history:
        if entry["role"] == "human":
            result.append(HumanMessage(content=entry["content"]))
        else:
            result.append(AIMessage(content=entry["content"]))
    return result


def _build_response_input(
    username: str, user_input: str, worker_output: str, context
) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [f"Текущая дата и время: {now}", ""]

    user_facts = (context or {}).get("user_facts") or {}
    if user_facts:
        parts.append("Что я знаю об участниках чата:")
        for uname, facts in user_facts.items():
            parts.append(f"@{uname}: {'; '.join(facts)}")
        parts.append("")

    recent = ((context or {}).get("recent_history") or [])[:RECENT_FILL_LIMIT]
    if recent:
        parts.append("Недавние сообщения чата:")
        for row in reversed(recent):
            parts.append(_render_row(row))
        parts.append("")

    replied_to = (context or {}).get("replied_to")
    if replied_to:
        recent_ids = {row["message_id"] for row in recent}
        if replied_to["message_id"] not in recent_ids:
            parts.append("Сообщение, на которое отвечают:")
            parts.append(_render_row(replied_to))
            parts.append("")

    if worker_output:
        parts.append(f"[Собранные данные]:\n{worker_output}\n")

    if replied_to:
        trigger = f"@{username} (↳ @{replied_to['username']}): {user_input}"
    else:
        trigger = f"@{username}: {user_input}"
    parts.append(trigger)

    return "\n".join(parts)


class ResponseNode:
    """Generates the final personality-driven reply from gathered worker facts."""

    def __init__(self, agent) -> None:
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        thread_id = state.get("thread_id") or str(msg["chat_id"])

        history = await thread_history.get_history(
            thread_id=thread_id, limit=config.MAX_HISTORY_MESSAGES
        )
        past_messages = _build_past_messages(history)

        user_input = msg["processed_text"] or msg["raw_text"] or ""
        enriched = _build_response_input(
            msg["username"],
            user_input,
            state.get("worker_output") or "",
            state.get("context"),
        )
        ai_message = await self.__generate(past_messages, enriched)
        response_text = strip_thinking(ai_message.content) if ai_message else ""

        if response_text.strip():
            await thread_history.append_turn(
                thread_id=thread_id,
                chat_id=msg["chat_id"],
                human_content=f"@{msg['username']}: {user_input}",
                ai_content=strip_markdown(response_text),
            )

        return {"response": response_text}

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
