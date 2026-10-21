"""
ResponseNode tests.

Scenario from 4a6cd4e: the full enriched prompt (including reply chains,
recent history, worker output) was stored as the human turn in thread_history.
When the user then replied to a different message, the LLM saw the old enriched
context as the most recent human message and responded about the wrong thread.

Fix: only "@username: user_input" is stored — context is assembled fresh each turn.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm import ContextLengthError, DailyLimitError, RateLimitError
from src.pipeline.response_node import ResponseNode, log_response_usage, no_text_reply
from src.pipeline.state import WorkerStatus
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

    async def test_stored_ai_turn_has_leaked_speaker_label_stripped(self):
        """The response model sometimes continues the history transcript it was
        shown instead of answering, echoing its own "Ты (бот):" label back.
        That label must never be persisted, or future turns learn to repeat it."""
        agent = make_mock_agent(response_text="Ты (бот): Ну да, но не признаюсь.")

        incoming = make_incoming(username="bob", raw_text="ты еблан?", processed_text="ты еблан?")
        state = make_state(
            incoming,
            should_respond=True,
            thread_id="thread-100",
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
        assert stored_ai_content == "Ну да, но не признаюсь."


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


class TestNoTextReply:
    """The meme (or the generated group profile) is the whole reply — no LLM call, no
    worker facts, and nothing to stack a comment on."""

    @pytest.mark.parametrize("flag", ["meme_request", "group_profile_request"])
    async def test_no_text_reply_matches_the_old_skipped_path(self, flag):
        result = await no_text_reply(make_state(make_incoming(), **{flag: True}))
        assert result == {
            "worker_output": "", "search_notification_msg": None, "worker_tools_used": False,
            "worker_status": WorkerStatus.SKIPPED,
            "response": "", "response_messages": [], "response_trace": None,
        }


class TestOrdinaryMessageGeneration:
    async def test_ordinary_message_still_generates(self):
        agent = make_mock_agent("обычный ответ")
        state = make_state(make_incoming(), is_flat_thread=True)
        with patch(THREAD_APPEND_TURN, AsyncMock()):
            result = await ResponseNode(agent)(state)
        assert result["response"] == "обычный ответ"


class TestLogResponseUsage:
    """Logs the real per-call token usage Groq already returns — no local
    tokenizer, no estimate, no third-party network call (see the function's
    own docstring for why tiktoken/get_num_tokens were rejected).
    """

    def test_logs_real_usage_at_debug(self, caplog):
        caplog.set_level(logging.DEBUG)
        log_response_usage({"input_tokens": 1800, "output_tokens": 40, "total_tokens": 1840})
        assert "input=1800" in caplog.text
        assert "output=40" in caplog.text
        assert "total=1840" in caplog.text

    def test_noop_when_debug_disabled(self, caplog):
        caplog.set_level(logging.INFO)
        log_response_usage({"input_tokens": 1800, "output_tokens": 40, "total_tokens": 1840})
        assert "input=" not in caplog.text

    def test_noop_when_usage_is_empty(self, caplog):
        """The model can return no usage_metadata at all (see usage_sink in
        ResponseAgent.invoke_response) — nothing should be logged."""
        caplog.set_level(logging.DEBUG)
        log_response_usage({})
        assert "input=" not in caplog.text


class TestResponseTrace:
    async def test_builds_trace_with_prompt_and_response(self):
        agent = make_mock_agent("Это ответ бота.")

        async def fake_invoke(messages, usage_sink=None):
            usage_sink.update({"input_tokens": 50, "output_tokens": 4, "model_name": "test-model"})
            return "Это ответ бота."

        agent.invoke_response = AsyncMock(side_effect=fake_invoke)
        node = ResponseNode(agent)
        incoming = make_incoming(username="vasya", raw_text="привет")
        state = make_state(incoming, response_trigger="explicit", filter_verdict="MEANINGFUL")
        with patch(THREAD_GET_HISTORY, AsyncMock(return_value=[])), \
             patch(THREAD_APPEND_TURN, AsyncMock()):
            result = await node(state)

        trace = result["response_trace"]
        assert trace["response"] == "Это ответ бота."
        assert trace["raw_response"] == "Это ответ бота."
        assert trace["model"] == "test-model"
        assert trace["input_tokens"] == 50
        assert trace["output_tokens"] == 4
        assert trace["language_corrected"] is False
        assert isinstance(trace["latency_ms"], int)
        assert trace["user_prompt"]  # the enriched final turn, non-empty
        assert trace["history_messages"] == []
