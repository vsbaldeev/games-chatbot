"""Tests for agent utility functions: strip_thinking, apply_language_correction,
should_retry, GroqContextGuard, and Agent.

Regression for the Qwen3 thinking-block leak: the model emits <think>...</think>
in its raw output and the bot was forwarding that entire block to the user instead
of the actual answer.

Fix: strip_thinking() removes thinking blocks at every consumption point.
     apply_language_correction() checks only the visible answer for foreign script
     so thinking-internal foreign text doesn't trigger a spurious Russian retry.
"""

from unittest.mock import AsyncMock, MagicMock

import groq
import pytest

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage as LCAIMessage

from src.agent import (
    ContextLengthError,
    DailyLimitError,
    GroqContextGuard,
    RateLimitError,
    ResponseAgent,
    ThinkingStripper,
    WorkerAgent,
    apply_language_correction,
    should_retry,
    strip_thinking,
)
from tests.builders import make_bad_request_error, make_rate_limit_error


class TestStripThinking:
    def test_text_without_tags_returned_unchanged(self):
        assert strip_thinking("Через 191 день.") == "Через 191 день."

    def test_single_think_block_removed(self):
        text = "<think>some internal reasoning</think>Через 191 день."
        assert strip_thinking(text) == "Через 191 день."

    def test_multiline_think_block_removed(self):
        text = "<think>\nline one\nline two\n</think>Ответ."
        assert strip_thinking(text) == "Ответ."

    def test_only_think_block_gives_empty_string(self):
        assert strip_thinking("<think>just thinking</think>") == ""

    def test_empty_string_stays_empty(self):
        assert strip_thinking("") == ""

    def test_surrounding_whitespace_stripped(self):
        assert strip_thinking("<think>reasoning</think>  Ответ.  ") == "Ответ."

    def test_tag_matching_is_case_insensitive(self):
        assert strip_thinking("<THINK>reasoning</THINK>Результат.") == "Результат."


class TestApplyLanguageCorrection:
    async def test_clean_russian_answer_returns_message_unchanged(self):
        llm = MagicMock()
        ai_message = MagicMock()
        ai_message.content = "Это нормальный русский ответ."
        result = await apply_language_correction(llm, ai_message, [])
        assert result is ai_message
        llm.ainvoke.assert_not_called()

    async def test_foreign_script_inside_think_block_does_not_trigger_retry(self):
        """Reasoning steps may contain foreign words while thinking — only the
        visible answer outside the think block should trigger the language retry."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        ai_message = MagicMock()
        ai_message.content = "<think>こんにちは、考えています</think>Нормальный ответ."
        result = await apply_language_correction(llm, ai_message, [])
        assert result is ai_message
        llm.ainvoke.assert_not_called()

    async def test_foreign_script_in_visible_answer_triggers_retry(self):
        """When the actual answer outside the think block contains foreign script,
        the Russian-language retry must fire."""
        corrected = MagicMock()
        corrected.content = "Исправленный ответ."
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=corrected)
        ai_message = MagicMock()
        ai_message.content = "<think>normal thinking</think>こんにちは"
        result = await apply_language_correction(llm, ai_message, [])
        assert result is corrected

    async def test_correction_llm_error_falls_back_to_original(self):
        """If the correction LLM call itself fails, return the original message
        rather than crashing the pipeline."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        ai_message = MagicMock()
        ai_message.content = "こんにちは"
        result = await apply_language_correction(llm, ai_message, [])
        assert result is ai_message


class TestShouldRetry:
    def test_transient_rate_limit_returns_true(self):
        err = make_rate_limit_error("rate_limit_exceeded: 429 too many requests per minute")
        assert should_retry(err) is True

    def test_daily_quota_returns_false(self):
        err = make_rate_limit_error("rate_limit_exceeded: 429 tokens per day exceeded")
        assert should_retry(err) is False

    def test_daily_keyword_returns_false(self):
        err = make_rate_limit_error("daily limit exhausted")
        assert should_retry(err) is False

    def test_tokens_per_day_keyword_returns_false(self):
        err = make_rate_limit_error("tokens_per_day quota exceeded")
        assert should_retry(err) is False

    def test_non_rate_limit_error_returns_false(self):
        assert should_retry(Exception("something else")) is False

    def test_bad_request_error_returns_false(self):
        err = make_bad_request_error("request too large")
        assert should_retry(err) is False


class TestGroqContextGuard:
    async def test_happy_path_passes_result_through(self):
        guard = GroqContextGuard()
        expected = object()
        handler = AsyncMock(return_value=expected)
        result = await guard.awrap_model_call(None, handler)
        assert result is expected

    @pytest.mark.parametrize("phrase", [
        "context_length_exceeded",
        "request too large",
        "maximum context length",
        "input too long",
        "tokens_in_context",
    ])
    async def test_context_length_bad_request_raises_context_length_error(self, phrase):
        guard = GroqContextGuard()
        err = make_bad_request_error(phrase)
        handler = AsyncMock(side_effect=err)
        with pytest.raises(ContextLengthError):
            await guard.awrap_model_call(None, handler)

    async def test_non_context_bad_request_passes_through(self):
        """A 400 error that is not about context length (e.g. bad tool schema)
        must not be converted — it should propagate as-is."""
        guard = GroqContextGuard()
        err = make_bad_request_error("invalid tool definition")
        handler = AsyncMock(side_effect=err)
        with pytest.raises(groq.BadRequestError):
            await guard.awrap_model_call(None, handler)

    async def test_non_groq_error_passes_through(self):
        guard = GroqContextGuard()
        handler = AsyncMock(side_effect=ValueError("unrelated"))
        with pytest.raises(ValueError):
            await guard.awrap_model_call(None, handler)


class TestWorkerAgent:
    async def test_invoke_worker_raises_before_init(self):
        """invoke_worker must fail immediately when no executor has been built."""
        with pytest.raises(RuntimeError):
            await WorkerAgent().invoke_worker("prompt")


class TestResponseAgent:
    async def test_invoke_response_raises_before_init(self):
        """invoke_response must fail immediately when no executor has been built."""
        with pytest.raises(RuntimeError):
            await ResponseAgent().invoke_response([])


def make_worker_agent(*, content=None, error=None):
    """Return a WorkerAgent with an injected mock executor."""
    last_message = MagicMock()
    last_message.content = content or "output"
    executor = MagicMock()
    if error is not None:
        executor.ainvoke = AsyncMock(side_effect=error)
    else:
        executor.ainvoke = AsyncMock(return_value={"messages": [last_message]})
    return WorkerAgent(worker_executor=executor)


def make_response_agent(*, content=None, error=None):
    """Return a ResponseAgent with an injected mock executor."""
    last_message = MagicMock()
    last_message.content = content or "ответ"
    executor = MagicMock()
    if error is not None:
        executor.ainvoke = AsyncMock(side_effect=error)
    else:
        executor.ainvoke = AsyncMock(return_value={"messages": [last_message]})
    return ResponseAgent(response_executor=executor)


class TestThinkingStripper:
    async def test_strips_think_block_from_last_message(self):
        """aafter_model must strip <think>...</think> from the last AI message."""
        raw = "<think>внутренние рассуждения</think>GTA 6 выходит 19 ноября 2026."
        state = {"messages": [LCAIMessage(content=raw)]}
        result = await ThinkingStripper().aafter_model(state, None)
        assert result is not None
        assert result["messages"][-1].content == "GTA 6 выходит 19 ноября 2026."

    async def test_returns_none_when_no_think_block(self):
        """aafter_model returns None (no change) when content has no think block."""
        state = {"messages": [LCAIMessage(content="Через 191 день.")]}
        result = await ThinkingStripper().aafter_model(state, None)
        assert result is None

    async def test_preserves_preceding_messages(self):
        """Only the last message is replaced; earlier messages stay intact."""
        earlier = LCAIMessage(content="ранее")
        last = LCAIMessage(content="<think>мысли</think>Ответ.")
        state = {"messages": [earlier, last]}
        result = await ThinkingStripper().aafter_model(state, None)
        assert result["messages"][0].content == "ранее"
        assert result["messages"][1].content == "Ответ."

    async def test_returns_none_for_empty_message_list(self):
        """aafter_model handles an empty message list without raising."""
        result = await ThinkingStripper().aafter_model({"messages": []}, None)
        assert result is None


class TestInvokeWorker:
    async def test_returns_executor_output(self):
        """invoke_worker returns the last message content from the executor result."""
        result = await make_worker_agent(content="GTA 6 выходит 19 ноября 2026.").invoke_worker("когда выйдет GTA 6?")
        assert "GTA 6 выходит 19 ноября 2026." in result

    async def test_context_length_error_propagates(self):
        """ContextLengthError raised by the executor must propagate unchanged."""
        with pytest.raises(ContextLengthError):
            await make_worker_agent(error=ContextLengthError("too long")).invoke_worker("prompt")

    async def test_daily_limit_maps_to_daily_limit_error(self):
        """Groq 429 with a daily-quota phrase must become DailyLimitError."""
        with pytest.raises(DailyLimitError):
            await make_worker_agent(error=make_rate_limit_error("per day limit exhausted")).invoke_worker("prompt")

    async def test_transient_rate_limit_maps_to_rate_limit_error(self):
        """Groq 429 without a daily phrase must become RateLimitError."""
        with pytest.raises(RateLimitError):
            await make_worker_agent(error=make_rate_limit_error("too many requests per minute")).invoke_worker("prompt")


class TestInvokeResponse:
    async def test_returns_llm_content(self):
        """invoke_response returns the text from the LLM reply."""
        result = await make_response_agent(content="Через 191 день.").invoke_response([HumanMessage(content="вопрос")])
        assert "Через 191 день." in result

    async def test_context_length_error_maps(self):
        """Groq 400 with a context-length phrase must become ContextLengthError."""
        with pytest.raises(ContextLengthError):
            await make_response_agent(error=make_bad_request_error("context_length_exceeded")).invoke_response([HumanMessage(content="вопрос")])

    async def test_daily_limit_maps_to_daily_limit_error(self):
        """Groq 429 with a daily-quota phrase must become DailyLimitError."""
        with pytest.raises(DailyLimitError):
            await make_response_agent(error=make_rate_limit_error("per day limit exhausted")).invoke_response([HumanMessage(content="вопрос")])

    async def test_transient_rate_limit_maps_to_rate_limit_error(self):
        """Groq 429 without a daily phrase must become RateLimitError."""
        with pytest.raises(RateLimitError):
            await make_response_agent(error=make_rate_limit_error("too many requests per minute")).invoke_response([HumanMessage(content="вопрос")])

    async def test_non_context_bad_request_propagates(self):
        """A 400 error unrelated to context length must propagate as BadRequestError."""
        with pytest.raises(groq.BadRequestError):
            await make_response_agent(error=make_bad_request_error("invalid tool definition")).invoke_response([HumanMessage(content="вопрос")])
