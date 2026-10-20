"""Tests for WorkerAgent and ResponseAgent: executor output, exception mapping, usage sink."""

from unittest.mock import AsyncMock, MagicMock

import groq
import pytest
from langchain_core.messages import HumanMessage

from src.agent import ContextLengthError, DailyLimitError, RateLimitError
from src.pipeline.agents import ResponseAgent, WorkerAgent
from tests.builders import make_bad_request_error, make_rate_limit_error


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


def make_response_agent(*, content=None, error=None, usage_metadata=None, model_name=None):
    """Return a ResponseAgent with an injected mock executor."""
    last_message = MagicMock()
    last_message.content = content or "ответ"
    last_message.usage_metadata = usage_metadata
    last_message.response_metadata = {"model_name": model_name} if model_name else {}
    executor = MagicMock()
    if error is not None:
        executor.ainvoke = AsyncMock(side_effect=error)
    else:
        executor.ainvoke = AsyncMock(return_value={"messages": [last_message]})
    return ResponseAgent(response_executor=executor)


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


class TestInvokeResponseUsageSink:
    """usage_sink lets callers read the real Groq token usage for the DEBUG
    per-block estimate comparison (response_node.log_response_usage) without
    changing invoke_response's return contract for existing callers."""

    async def test_usage_sink_is_populated_when_present(self):
        agent = make_response_agent(
            content="ответ",
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        )
        usage_sink: dict = {}

        await agent.invoke_response([HumanMessage(content="вопрос")], usage_sink=usage_sink)

        assert usage_sink == {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}

    async def test_usage_sink_stays_empty_when_model_reports_none(self):
        agent = make_response_agent(content="ответ", usage_metadata=None)
        usage_sink: dict = {}

        await agent.invoke_response([HumanMessage(content="вопрос")], usage_sink=usage_sink)

        assert usage_sink == {}

    async def test_usage_sink_is_optional(self):
        """Existing callers that never pass usage_sink must be unaffected."""
        result = await make_response_agent(content="ответ").invoke_response([HumanMessage(content="вопрос")])
        assert result == "ответ"

    async def test_model_name_captured_alongside_usage(self):
        """The model that actually answered (post-fallback) is captured for
        bot_llm_log — see src.store.llm_log.insert_call."""
        agent = make_response_agent(
            content="ответ",
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
            model_name="openai/gpt-oss-120b",
        )
        usage_sink: dict = {}

        await agent.invoke_response([HumanMessage(content="вопрос")], usage_sink=usage_sink)

        assert usage_sink["model_name"] == "openai/gpt-oss-120b"

    async def test_model_name_absent_from_sink_when_not_reported(self):
        agent = make_response_agent(content="ответ", usage_metadata=None)
        usage_sink: dict = {}

        await agent.invoke_response([HumanMessage(content="вопрос")], usage_sink=usage_sink)

        assert "model_name" not in usage_sink
