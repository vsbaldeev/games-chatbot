"""
ResponseNode tests.

Scenario from 4a6cd4e: the full enriched prompt (including reply chains,
recent history, worker output) was stored as the human turn in thread_history.
When the user then replied to a different message, the LLM saw the old enriched
context as the most recent human message and responded about the wrong thread.

Fix: only "@username: user_input" is stored — context is assembled fresh each turn.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipeline.response_node import ResponseNode
from tests.builders import make_incoming, make_state

THREAD_GET_HISTORY = "src.pipeline.response_node.thread_history.get_history"
THREAD_APPEND_TURN = "src.pipeline.response_node.thread_history.append_turn"
APPLY_LANGUAGE_CORRECTION = "src.pipeline.response_node.apply_language_correction"


def make_mock_agent(response_text: str = "Это ответ бота.") -> MagicMock:
    ai_message = MagicMock()
    ai_message.content = response_text
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=ai_message)
    agent = MagicMock()
    agent.get_response_llm.return_value = llm
    return agent


class TestThreadHistoryStorage:
    async def test_stored_human_turn_is_only_username_and_text(self):
        """
        Regression (4a6cd4e): history stored the full enriched prompt with embedded
        reply chains. Next turn the LLM read that as fresh context and answered
        about the wrong thread.  Only the bare user utterance must be persisted.
        """
        ai_response = MagicMock()
        ai_response.content = "Это ответ бота."
        agent = make_mock_agent()

        incoming = make_incoming(
            username="alice",
            raw_text="что в этом фото?",
            processed_text="что в этом фото?",
        )
        state = make_state(
            incoming,
            should_respond=True,
            thread_id="thread-42",
            context={
                "user_facts": {},
                "recent_history": [{"message_id": 1, "username": "bob", "content": "hi", "media_type": "text", "user_id": 2}],
                "replied_to": {"message_id": 1, "username": "bob", "content": "hi", "media_type": "text", "user_id": 2},
                "reply_chain": [{"message_id": 1, "username": "bob", "content": "hi", "media_type": "text", "user_id": 2}],
            },
            worker_output="[Данные]: некая информация об игре",
        )

        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock) as mock_append,
            patch(APPLY_LANGUAGE_CORRECTION, new_callable=AsyncMock, return_value=ai_response),
        ):
            response_node = ResponseNode(agent)
            await response_node(state)

        mock_append.assert_called_once()
        stored_human_content = mock_append.call_args.kwargs["human_content"]

        assert stored_human_content == "@alice: что в этом фото?"
        assert "reply_chain" not in stored_human_content
        assert "Собранные данные" not in stored_human_content
        assert "Недавние сообщения" not in stored_human_content
        assert "worker_output" not in stored_human_content

    async def test_stored_ai_turn_is_stripped_of_markdown(self):
        """Markdown in AI responses must be stripped before storage to prevent
        the LLM from reinforcing its own markdown formatting in future turns."""
        ai_response = MagicMock()
        ai_response.content = "**Жирный** и _курсив_ текст."
        agent = make_mock_agent()
        agent.get_response_llm.return_value.ainvoke = AsyncMock(return_value=ai_response)

        incoming = make_incoming(username="bob", raw_text="расскажи", processed_text="расскажи")
        state = make_state(
            incoming,
            should_respond=True,
            thread_id="thread-99",
            context={"user_facts": {}, "recent_history": [], "replied_to": None, "reply_chain": []},
            worker_output="",
        )

        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock) as mock_append,
            patch(APPLY_LANGUAGE_CORRECTION, new_callable=AsyncMock, return_value=ai_response),
        ):
            response_node = ResponseNode(agent)
            await response_node(state)

        stored_ai_content = mock_append.call_args.kwargs["ai_content"]
        assert "**" not in stored_ai_content
        assert stored_ai_content == "Жирный и курсив текст."


class TestThinkingBlockStripping:
    async def test_think_block_not_sent_to_user(self):
        """Reasoning models (e.g. Qwen3) emit <think>...</think> before the answer.
        The user must never see internal reasoning traces — only the final answer."""
        ai_response = MagicMock()
        ai_response.content = (
            "<think>Пользователь спрашивает про GTA 6. "
            "Дата релиза 19 ноября 2026, сегодня 12 мая 2026, разница 191 день.</think>"
            "Через 191 день."
        )
        agent = make_mock_agent()
        agent.get_response_llm.return_value.ainvoke = AsyncMock(return_value=ai_response)

        incoming = make_incoming(
            username="bob",
            raw_text="через сколько дней GTA 6?",
            processed_text="через сколько дней GTA 6?",
        )
        state = make_state(
            incoming,
            should_respond=True,
            thread_id="thread-42",
            context={"user_facts": {}, "recent_history": [], "replied_to": None, "reply_chain": []},
            worker_output="GTA 6 выходит 19 ноября 2026. Сегодня 12 мая 2026. До релиза 191 день.",
        )

        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
            patch(APPLY_LANGUAGE_CORRECTION, new_callable=AsyncMock, return_value=ai_response),
        ):
            result = await ResponseNode(agent)(state)

        assert "<think>" not in result["response"]
        assert "Через 191 день." in result["response"]
