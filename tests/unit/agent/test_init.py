"""Tests for agent utility functions: strip_thinking, apply_language_correction,
should_retry, GroqContextGuard, and Agent.

Regression for the Qwen3 thinking-block leak: the model emits <think>...</think>
in its raw output and the bot was forwarding that entire block to the user instead
of the actual answer.

Fix: strip_thinking() removes thinking blocks at every consumption point.
     apply_language_correction() checks only the visible answer for foreign script
     so thinking-internal foreign text doesn't trigger a spurious Russian retry.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import groq
import pytest

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.agent import (
    ContextLengthError,
    DailyLimitError,
    GroqContextGuard,
    ModelAttemptLogger,
    RateLimitError,
    ThinkingStripper,
    apply_language_correction,
    guarded_ainvoke,
    should_retry,
    strip_thinking,
)
from tests.builders import (
    make_bad_request_error,
    make_openai_rate_limit_error,
    make_rate_limit_error,
)


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

    def test_openrouter_transient_rate_limit_returns_true(self):
        """The response/roast chains' OpenRouter fallback leg raises
        openai.RateLimitError, not groq.RateLimitError — must be retried too."""
        err = make_openai_rate_limit_error("rate-limited upstream")
        assert should_retry(err) is True

    def test_openrouter_daily_quota_returns_false(self):
        err = make_openai_rate_limit_error("tokens_per_day quota exceeded")
        assert should_retry(err) is False


class TestGuardedAinvoke:
    async def test_openrouter_rate_limit_maps_to_rate_limit_error(self):
        """An exhausted OpenRouter 429 must surface as the typed RateLimitError,
        same as a Groq one, so the pipeline sends the rate-limit notice instead
        of logging a raw traceback and the generic failure notice."""
        runnable = MagicMock()
        runnable.ainvoke = AsyncMock(side_effect=make_openai_rate_limit_error("rate-limited upstream"))
        with pytest.raises(RateLimitError):
            await guarded_ainvoke(runnable, {"messages": []})

    async def test_openrouter_daily_quota_maps_to_daily_limit_error(self):
        runnable = MagicMock()
        runnable.ainvoke = AsyncMock(
            side_effect=make_openai_rate_limit_error("tokens_per_day quota exceeded")
        )
        with pytest.raises(DailyLimitError):
            await guarded_ainvoke(runnable, {"messages": []})


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


def make_model_request(model, messages=None) -> SimpleNamespace:
    return SimpleNamespace(model=model, messages=messages if messages is not None else [])


class TestModelAttemptLogger:
    """Neither ModelFallbackMiddleware nor ModelRetryMiddleware log which
    provider/model a call actually hit — this middleware fills that gap so
    a fallover between Groq and OpenRouter (or between OpenRouter models) is
    visible in the logs instead of just a stream of unattributed 429s, along
    with what was actually sent and returned."""

    async def test_successful_openrouter_call_logs_provider_and_model_at_debug(self, caplog):
        request = make_model_request(
            ChatOpenAI(model="z-ai/glm-5.2:free", api_key="x", base_url="https://openrouter.ai/api/v1")
        )
        handler = AsyncMock(return_value=SimpleNamespace(result=[LCAIMessage(content="привет")]))
        with caplog.at_level(logging.DEBUG, logger="src.agent.middleware"):
            result = await ModelAttemptLogger().awrap_model_call(request, handler)
        assert result.result[0].content == "привет"
        assert "openrouter.ai/z-ai/glm-5.2:free" in caplog.text

    async def test_failed_groq_call_logs_provider_and_model_at_warning_then_reraises(self, caplog):
        request = make_model_request(ChatGroq(model="openai/gpt-oss-120b", api_key="x"))
        handler = AsyncMock(side_effect=ValueError("boom"))
        with caplog.at_level(logging.WARNING, logger="src.agent.middleware"), pytest.raises(ValueError):
            await ModelAttemptLogger().awrap_model_call(request, handler)
        assert "groq.com/openai/gpt-oss-120b" in caplog.text

    async def test_distinguishes_openrouter_models_sharing_the_chatopenai_class(self, caplog):
        """Two OpenRouter fallback legs (Gemma, GLM) are both ChatOpenAI instances —
        the log must still show which specific model was hit, not just the class."""
        gemma_request = make_model_request(
            ChatOpenAI(model="google/gemma-4-31b-it:free", api_key="x", base_url="https://openrouter.ai/api/v1")
        )
        handler = AsyncMock(return_value=SimpleNamespace(result=[LCAIMessage(content="ok")]))
        with caplog.at_level(logging.DEBUG, logger="src.agent.middleware"):
            await ModelAttemptLogger().awrap_model_call(gemma_request, handler)
        assert "google/gemma-4-31b-it:free" in caplog.text
        assert "z-ai/glm-5.2:free" not in caplog.text

    async def test_logs_input_messages_on_success(self, caplog):
        request = make_model_request(
            ChatGroq(model="openai/gpt-oss-120b", api_key="x"),
            messages=[HumanMessage(content="когда выйдет GTA 6?")],
        )
        handler = AsyncMock(return_value=SimpleNamespace(result=[LCAIMessage(content="19 ноября 2026")]))
        with caplog.at_level(logging.DEBUG, logger="src.agent.middleware"):
            await ModelAttemptLogger().awrap_model_call(request, handler)
        assert "когда выйдет GTA 6?" in caplog.text
        assert "19 ноября 2026" in caplog.text

    async def test_logs_input_messages_on_failure(self, caplog):
        request = make_model_request(
            ChatGroq(model="openai/gpt-oss-120b", api_key="x"),
            messages=[HumanMessage(content="что там по игре?")],
        )
        handler = AsyncMock(side_effect=ValueError("boom"))
        with caplog.at_level(logging.WARNING, logger="src.agent.middleware"), pytest.raises(ValueError):
            await ModelAttemptLogger().awrap_model_call(request, handler)
        assert "что там по игре?" in caplog.text

    async def test_tool_call_output_shows_tool_names_when_content_is_empty(self, caplog):
        """A worker-agent turn requesting a tool has no text content — the log
        must still show something useful instead of an empty output=<empty>."""
        request = make_model_request(
            ChatGroq(model="openai/gpt-oss-120b", api_key="x"),
            messages=[HumanMessage(content="кто разработчик GTA 6?")],
        )
        tool_call_message = LCAIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "GTA 6 developer"}, "id": "1"}],
        )
        handler = AsyncMock(return_value=SimpleNamespace(result=[tool_call_message]))
        with caplog.at_level(logging.DEBUG, logger="src.agent.middleware"):
            await ModelAttemptLogger().awrap_model_call(request, handler)
        assert "web_search" in caplog.text


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
