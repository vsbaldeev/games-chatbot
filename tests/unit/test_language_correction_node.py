"""Tests for LanguageCorrectionNode."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent import DailyLimitError, RateLimitError
from src.config.prompts import LANGUAGE_CORRECTION_PROMPT
from src.pipeline.language_correction_node import LanguageCorrectionNode


@pytest.fixture(autouse=True)
def stub_persist_thread_turn(monkeypatch):
    """Stub the post-correction history write — these tests use minimal states."""
    stub = AsyncMock()
    monkeypatch.setattr(
        "src.pipeline.language_correction_node.persist_thread_turn", stub
    )
    return stub


def make_node(*, response=None, error=None):
    """Return a LanguageCorrectionNode with an injected mock agent."""
    agent = MagicMock()
    if error is not None:
        agent.invoke_response = AsyncMock(side_effect=error)
    else:
        agent.invoke_response = AsyncMock(return_value=response or "Исправленный ответ.")
    return LanguageCorrectionNode(agent)


class TestLanguageCorrectionNode:
    async def test_returns_corrected_response(self):
        node = make_node(response="Исправленный ответ.")
        result = await node({"response_messages": [], "response": "こんにちは"})
        assert result == {"response": "Исправленный ответ."}

    async def test_correction_prompt_appended_to_messages(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(return_value="Ответ.")
        node = LanguageCorrectionNode(agent)
        original_messages = [SystemMessage(content="system")]
        await node({"response_messages": original_messages, "response": "こんにちは"})
        called_with = agent.invoke_response.call_args[0][0]
        assert called_with[0] is original_messages[0]
        assert isinstance(called_with[-1], HumanMessage)
        assert called_with[-1].content == LANGUAGE_CORRECTION_PROMPT

    async def test_original_messages_not_mutated(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(return_value="Ответ.")
        node = LanguageCorrectionNode(agent)
        original_messages = [HumanMessage(content="вопрос")]
        await node({"response_messages": original_messages, "response": "こんにちは"})
        assert len(original_messages) == 1

    async def test_none_response_messages_sends_only_correction_prompt(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(return_value="Ответ.")
        node = LanguageCorrectionNode(agent)
        await node({"response_messages": None, "response": "こんにちは"})
        called_with = agent.invoke_response.call_args[0][0]
        assert len(called_with) == 1
        assert called_with[0].content == LANGUAGE_CORRECTION_PROMPT

    async def test_daily_limit_error_propagates(self):
        node = make_node(error=DailyLimitError("quota"))
        with pytest.raises(DailyLimitError):
            await node({"response_messages": [], "response": "こんにちは"})

    async def test_rate_limit_error_propagates(self):
        node = make_node(error=RateLimitError("rate"))
        with pytest.raises(RateLimitError):
            await node({"response_messages": [], "response": "こんにちは"})

    async def test_generic_error_returns_empty_dict(self):
        node = make_node(error=Exception("oops"))
        result = await node({"response_messages": [], "response": "こんにちは"})
        assert result == {}
