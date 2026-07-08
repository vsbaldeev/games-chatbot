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

from src.agent import ContextLengthError, DailyLimitError, RateLimitError
from src.pipeline.response_node import (
    ResponseNode,
    build_asking_user_tag_lines,
    build_response_input,
)
from tests.builders import make_incoming, make_state

THREAD_GET_HISTORY = "src.pipeline.response_node.thread_history.get_history"
THREAD_APPEND_TURN = "src.pipeline.response_node.thread_history.append_turn"


def make_mock_agent(response_text: str = "Это ответ бота.") -> MagicMock:
    """Return a mock Agent whose invoke_response returns response_text."""
    agent = MagicMock()
    agent.invoke_response = AsyncMock(return_value=response_text)
    return agent


class TestThreadHistoryStorage:
    async def test_stored_human_turn_is_only_username_and_text(self):
        """
        Regression (4a6cd4e): history stored the full enriched prompt with embedded
        reply chains. Next turn the LLM read that as fresh context and answered
        about the wrong thread.  Only the bare user utterance must be persisted.
        """
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
        agent = make_mock_agent(response_text="**Жирный** и _курсив_ текст.")

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
        ):
            response_node = ResponseNode(agent)
            await response_node(state)

        stored_ai_content = mock_append.call_args.kwargs["ai_content"]
        assert "**" not in stored_ai_content
        assert stored_ai_content == "Жирный и курсив текст."


class TestThinkingBlockStripping:
    async def test_response_text_passed_through_from_agent(self):
        """ResponseNode places whatever invoke_response returns into state['response']
        without modification — thinking stripping is Agent's responsibility."""
        agent = make_mock_agent(response_text="Через 191 день.")

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
            worker_output="GTA 6 выходит 19 ноября 2026.",
        )

        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            result = await ResponseNode(agent)(state)

        assert result["response"] == "Через 191 день."


class TestRandomTriggerContext:
    """Random triggers get a thin recent-history slice (RANDOM_TRIGGER_CONTEXT_LIMIT).

    Root cause (46013): random photo trigger injected the full recent chat
    history into the response prompt, so the LLM addressed multiple open threads
    and broke the 'never suggest commands unprompted' rule. The revised contract
    keeps the newest few messages so the model can catch obvious topic mismatch
    without turning a spontaneous reaction into a reply to the discussion.
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

    def test_random_trigger_gets_reduced_history_slice(self):
        """Random triggers see only the newest RANDOM_TRIGGER_CONTEXT_LIMIT
        messages — older chat noise must stay out of the prompt."""
        recent_newest_first = [
            {
                "message_id": index,
                "username": "bob",
                "content": f"сообщение номер {index}",
                "media_type": "text",
                "user_id": 2,
            }
            for index in range(1, 5)
        ]
        context = {
            "user_facts": {},
            "recent_history": recent_newest_first,
            "replied_to": None,
            "reply_chain": [],
        }

        result = build_response_input(
            "tmaxims",
            "Изображение: руководство по стрижкам",
            "",
            context,
            response_trigger="random",
        )

        assert "сообщение номер 1" in result
        assert "сообщение номер 3" in result
        assert "сообщение номер 4" not in result

    def test_explicit_trigger_includes_recent_history(self):
        """When the user explicitly @mentioned the bot or replied to it, recent
        history must still be included for conversational context."""
        result = build_response_input(
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

        result = build_response_input(
            "alice",
            "Изображение: что-то",
            "",
            context,
            response_trigger="random",
        )

        assert "любит PS5" in result

    def test_random_trigger_includes_worker_output(self):
        """Worker facts must always reach the response LLM even for random triggers."""
        result = build_response_input(
            "alice",
            "Изображение: стрижки",
            "Руководство содержит 12 стилей.",
            self.make_context(),
            response_trigger="random",
        )

        assert "Руководство содержит 12 стилей." in result


class TestAskingUserTagLines:
    """The asker's weekly role + reason must be injected so the bot can explain it;
    nothing is emitted when the asker has no role."""

    def test_emits_role_and_reason_when_present(self):
        context = {
            "user_facts": {},
            "recent_history": [],
            "replied_to": None,
            "reply_chain": [],
            "asking_user_tag": {"tag": "Ночной дозор", "reason": "пишет после полуночи"},
        }
        lines = build_asking_user_tag_lines(context, "alice")
        joined = "\n".join(lines)
        assert "Ночной дозор" in joined
        assert "пишет после полуночи" in joined
        assert "alice" in joined

    def test_emits_nothing_when_absent(self):
        context = {"user_facts": {}, "recent_history": [], "replied_to": None, "reply_chain": []}
        assert build_asking_user_tag_lines(context, "alice") == []

    def test_build_response_input_includes_role_block(self):
        context = {
            "user_facts": {},
            "recent_history": [],
            "replied_to": None,
            "reply_chain": [],
            "asking_user_tag": {"tag": "Спидранер", "reason": "проходит за день"},
        }
        result = build_response_input(
            "alice", "почему у меня такая роль?", "", context, response_trigger="explicit"
        )
        assert "Спидранер" in result
        assert "проходит за день" in result


class TestErrorPropagation:
    """Typed exceptions raised by agent.invoke_response must propagate through
    ResponseNode so the top-level handler can surface them to the user."""

    def make_state_for_error(self):
        incoming = make_incoming(username="bob", raw_text="вопрос", processed_text="вопрос")
        return make_state(
            incoming,
            should_respond=True,
            context={"user_facts": {}, "recent_history": [], "replied_to": None, "reply_chain": []},
            worker_output="",
        )

    async def test_context_length_error_propagates(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(side_effect=ContextLengthError("too long"))
        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            with pytest.raises(ContextLengthError):
                await ResponseNode(agent)(self.make_state_for_error())

    async def test_daily_limit_error_propagates(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(side_effect=DailyLimitError("quota"))
        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            with pytest.raises(DailyLimitError):
                await ResponseNode(agent)(self.make_state_for_error())

    async def test_rate_limit_error_propagates(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(side_effect=RateLimitError("rate limit"))
        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            with pytest.raises(RateLimitError):
                await ResponseNode(agent)(self.make_state_for_error())

    async def test_unknown_exception_propagates_unchanged(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(side_effect=RuntimeError("unexpected failure"))
        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="unexpected failure"):
                await ResponseNode(agent)(self.make_state_for_error())
