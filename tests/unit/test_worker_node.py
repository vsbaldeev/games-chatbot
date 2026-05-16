"""WorkerNode tests.

Regression for the forwarded-article scenario: a user forwarding a post from
another channel, then replying with "@bot what's this article about?" triggered
a web_search even though the article text was already in the reply chain.

Root cause: WORKER_PROMPT had no rule about using existing context first.
Fix: CONTEXT FIRST clause added before TOOL SELECTION in agent.py.

Two invariants tested here:
  1. The prompt carries the CONTEXT FIRST rule in the right position.
  2. __build_worker_input includes reply-chain content in the assembled input
     so the rule has something to work with.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent import ContextLengthError, DailyLimitError, RateLimitError, WORKER_PROMPT
from src.pipeline.worker_node import WorkerNode
from tests.builders import make_incoming, make_message_row, make_state


def make_worker() -> WorkerNode:
    return WorkerNode(MagicMock())


def build_input(worker: WorkerNode, msg: dict, context: dict, response_trigger: str = "explicit") -> str:
    return worker._WorkerNode__build_worker_input(msg, context, response_trigger)


def make_worker_state(*, username: str = "alice", raw_text: str = "вопрос", processed_text: str = "вопрос"):
    incoming = make_incoming(username=username, raw_text=raw_text, processed_text=processed_text)
    return make_state(
        incoming,
        should_respond=True,
        context={"reply_chain": [], "recent_history": []},
    )


class TestWorkerPrompt:
    def test_context_first_rule_exists(self):
        """WORKER_PROMPT must contain the СНАЧАЛА КОНТЕКСТ directive so the
        LLM knows to use existing reply-chain content before reaching for tools."""
        assert "СНАЧАЛА КОНТЕКСТ" in WORKER_PROMPT

    def test_context_first_precedes_tool_selection(self):
        """СНАЧАЛА КОНТЕКСТ must appear before ВЫБОР ИНСТРУМЕНТА so the model evaluates
        available context before deciding whether a tool call is needed."""
        context_first_pos = WORKER_PROMPT.index("СНАЧАЛА КОНТЕКСТ")
        tool_selection_pos = WORKER_PROMPT.index("ВЫБОР ИНСТРУМЕНТА")
        assert context_first_pos < tool_selection_pos

    def test_prompt_forbids_tools_when_context_is_sufficient(self):
        """The prompt must explicitly say not to call tools when the needed
        content is already present in the reply chain."""
        assert "НЕ вызывай инструменты" in WORKER_PROMPT


class TestWorkerNodeBuildInput:
    """__build_worker_input assembles the LLM prompt string for the worker.

    The forwarded-article fix relies on this method correctly embedding the
    reply chain — without that, the CONTEXT FIRST directive has no content
    to point the model at.
    """

    def test_reply_chain_content_appears_in_worker_input(self):
        """Article text from a forwarded message must appear in the assembled
        input so the LLM can summarize it without calling web_search."""
        forwarded = make_message_row(
            message_id=10,
            username="news_channel",
            content="Подробный обзор новой игры с оценками и деталями релиза.",
        )
        incoming = make_incoming(
            username="alice",
            raw_text="@testbot что за статья?",
            processed_text="@testbot что за статья?",
        )
        context = {"reply_chain": [forwarded], "recent_history": []}

        result = build_input(make_worker(), incoming, context)

        assert "Подробный обзор новой игры с оценками и деталями релиза." in result

    def test_reply_chain_suppresses_recent_history(self):
        """When a reply chain is present, recent history must not appear in
        the input — they are mutually exclusive context sources."""
        chain_msg = make_message_row(message_id=20, username="fwd", content="контент из канала")
        recent_msg = make_message_row(message_id=5, username="bob", content="недавнее сообщение")
        incoming = make_incoming(
            username="alice",
            raw_text="@testbot объясни",
            processed_text="@testbot объясни",
        )
        context = {"reply_chain": [chain_msg], "recent_history": [recent_msg]}

        result = build_input(make_worker(), incoming, context)

        assert "контент из канала" in result
        assert "недавнее сообщение" not in result

    def test_recent_history_used_when_no_reply_chain(self):
        """Without a reply chain the worker falls back to recent history so
        conversational context is still available to the LLM."""
        recent_msg = make_message_row(message_id=5, username="bob", content="недавнее сообщение")
        incoming = make_incoming(
            username="alice",
            raw_text="@testbot помнишь?",
            processed_text="@testbot помнишь?",
        )
        context = {"reply_chain": [], "recent_history": [recent_msg]}

        result = build_input(make_worker(), incoming, context)

        assert "недавнее сообщение" in result

    def test_random_trigger_suppresses_recent_history_without_reply_chain(self):
        """When the bot randomly reacts to media (not @mentioned), unrelated recent
        chat history must be withheld so the worker focuses only on the media."""
        recent_msg = make_message_row(message_id=5, username="bob", content="посторонний разговор")
        incoming = make_incoming(
            username="tmaxims",
            raw_text=None,
            processed_text="Изображение: руководство по стрижкам",
            media_type="photo",
        )
        context = {"reply_chain": [], "recent_history": [recent_msg]}

        result = build_input(make_worker(), incoming, context, response_trigger="random")

        assert "посторонний разговор" not in result

    def test_reply_chain_included_for_random_trigger(self):
        """Reply chain is always the primary context source — it must be included
        even for random triggers because the user explicitly replied in that thread."""
        chain_msg = make_message_row(message_id=20, username="fwd", content="контент из цепочки")
        incoming = make_incoming(username="alice", raw_text="вопрос", processed_text="вопрос")
        context = {"reply_chain": [chain_msg], "recent_history": []}

        result = build_input(make_worker(), incoming, context, response_trigger="random")

        assert "контент из цепочки" in result

    def test_user_question_always_present(self):
        """The user's question must appear at the end of every worker input
        regardless of context availability."""
        incoming = make_incoming(
            username="alice",
            raw_text="@testbot что думаешь?",
            processed_text="@testbot что думаешь?",
        )
        context = {"reply_chain": [], "recent_history": []}

        result = build_input(make_worker(), incoming, context)

        assert "@alice" in result
        assert "@testbot что думаешь?" in result


class TestWorkerNodeOutput:
    """WorkerNode stores whatever invoke_worker returns in the pipeline state."""

    async def test_worker_output_stored_in_state(self):
        """Output from invoke_worker must be forwarded verbatim to worker_output."""
        agent = MagicMock()
        agent.invoke_worker = AsyncMock(return_value="GTA 6 выходит 19 ноября 2026.")
        state = make_worker_state(
            username="alice",
            raw_text="когда выйдет GTA 6?",
            processed_text="когда выйдет GTA 6?",
        )
        result = await WorkerNode(agent)(state)
        assert result["worker_output"] == "GTA 6 выходит 19 ноября 2026."


class TestContextLengthErrorHandling:
    """When invoke_worker raises ContextLengthError WorkerNode must absorb it
    and return empty output so the pipeline can still attempt a response."""

    async def test_context_length_error_returns_empty_output(self):
        agent = MagicMock()
        agent.invoke_worker = AsyncMock(side_effect=ContextLengthError("too long"))
        result = await WorkerNode(agent)(make_worker_state())
        assert result["worker_output"] == ""
        assert result["search_notification_msg"] is None


class TestErrorPropagation:
    """WorkerNode must propagate typed pipeline exceptions from invoke_worker
    so the top-level handler in events/messages.py can surface them to the user."""

    async def test_daily_limit_raises_daily_limit_error(self):
        agent = MagicMock()
        agent.invoke_worker = AsyncMock(side_effect=DailyLimitError("per day limit exhausted"))
        with pytest.raises(DailyLimitError):
            await WorkerNode(agent)(make_worker_state())

    async def test_transient_rate_limit_raises_rate_limit_error(self):
        agent = MagicMock()
        agent.invoke_worker = AsyncMock(side_effect=RateLimitError("rate_limit exceeded"))
        with pytest.raises(RateLimitError):
            await WorkerNode(agent)(make_worker_state())
