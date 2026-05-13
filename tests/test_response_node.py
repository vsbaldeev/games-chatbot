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

from src.agent import ContextLengthError
from src.pipeline.response_node import ResponseNode, _build_response_input
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


class TestRandomTriggerContext:
    """Regression for mixed-topic responses: when the bot randomly reacts to a
    photo it should not drag in unrelated recent-history threads.

    Root cause (46013): random photo trigger still injected the full recent chat
    history into the response prompt, so the LLM addressed multiple open threads
    and broke the 'never suggest commands unprompted' rule.
    """

    RECENT_MSG = {
        "message_id": 1,
        "username": "bob",
        "content": "я в мобильном",
        "media_type": "text",
        "user_id": 2,
    }

    def make_context(self) -> dict:
        return {
            "user_facts": {},
            "recent_history": [self.RECENT_MSG],
            "replied_to": None,
            "reply_chain": [],
        }

    def test_random_trigger_excludes_recent_history(self):
        """Unrelated recent chat must not appear in the prompt when the bot
        fired by random chance — prevents mixing unrelated conversation threads."""
        result = _build_response_input(
            "tmaxims",
            "Изображение: руководство по стрижкам",
            "",
            self.make_context(),
            response_trigger="random",
        )

        assert "я в мобильном" not in result

    def test_explicit_trigger_includes_recent_history(self):
        """When the user explicitly @mentioned the bot or replied to it, recent
        history must still be included for conversational context."""
        result = _build_response_input(
            "alice",
            "@bot что нового?",
            "",
            self.make_context(),
            response_trigger="explicit",
        )

        assert "я в мобильном" in result

    def test_random_trigger_still_includes_user_facts(self):
        """Per-user memories must always be injected — they personalise the
        response regardless of how the trigger fired."""
        context = {
            "user_facts": {"alice": ["любит PS5", "играет в FIFA"]},
            "recent_history": [self.RECENT_MSG],
            "replied_to": None,
            "reply_chain": [],
        }

        result = _build_response_input(
            "alice",
            "Изображение: что-то",
            "",
            context,
            response_trigger="random",
        )

        assert "любит PS5" in result
        assert "я в мобильном" not in result

    def test_random_trigger_includes_worker_output(self):
        """Worker facts must always reach the response LLM even for random triggers."""
        result = _build_response_input(
            "alice",
            "Изображение: стрижки",
            "Руководство содержит 12 стилей.",
            self.make_context(),
            response_trigger="random",
        )

        assert "Руководство содержит 12 стилей." in result


class TestContextLengthError:
    """When the response LLM rejects the prompt as too long, ResponseNode must
    raise ContextLengthError so the top-level handler can send a clear user message."""

    async def test_context_length_exception_raises_context_length_error(self):
        agent = MagicMock()
        agent.get_response_llm.return_value.ainvoke = AsyncMock(
            side_effect=Exception("context_length_exceeded: prompt too large")
        )

        incoming = make_incoming(username="bob", raw_text="вопрос", processed_text="вопрос")
        state = make_state(
            incoming,
            should_respond=True,
            context={"user_facts": {}, "recent_history": [], "replied_to": None, "reply_chain": []},
            worker_output="",
        )

        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            with pytest.raises(ContextLengthError):
                await ResponseNode(agent)(state)
