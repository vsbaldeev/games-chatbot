"""Respond subgraph routing-decision tests: guard, worker skip and foreign-script correction.
"""

from unittest.mock import MagicMock

from langgraph.graph import END
import pytest

from src.pipeline.respond.graph import (
    build_respond_graph,
    route_after_context,
    route_after_response,
    route_by_guard,
)
from src.pipeline.state import WorkerStatus
from tests.builders import make_incoming, make_state


class TestRouteAfterContext:
    """Replies that need only personality, or whose source material is already in the
    message, bypass the worker (and its «🔍 Ищу…» notification)."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"is_bot_insult": True},
            {"wind_down": True},
            {"response_trigger": "youtube_short"},
            {"response_trigger": "social_link"},
        ],
        ids=["bot-insult", "wind-down", "shorts", "social-link"],
    )
    def test_routes_directive_and_link_replies_past_the_worker(self, overrides):
        state = make_state(make_incoming(), should_respond=True, **overrides)
        assert route_after_context(state) == "skip_worker"

    def test_routes_an_ordinary_question_to_the_worker(self):
        state = make_state(make_incoming(), should_respond=True, response_trigger="explicit")
        assert route_after_context(state) == "worker"


class TestRouteByGuard:
    """A blocked message ends the run; a meme or group-profile request has no text reply
    to build, so it skips the context, worker and response nodes."""

    @pytest.mark.parametrize(
        "overrides, expected",
        [
            ({"blocked": True}, END),
            ({"meme_request": True}, "no_text_reply"),
            ({"group_profile_request": True}, "no_text_reply"),
            ({}, "context_builder"),
        ],
        ids=["blocked", "meme", "group-profile", "ordinary"],
    )
    def test_routes_by_block_and_request_flags(self, overrides, expected):
        state = make_state(make_incoming(), should_respond=True, **overrides)
        assert route_by_guard(state) == expected


class TestRouteAfterResponse:
    def test_clean_russian_reply_ends_the_subgraph(self):
        state = make_state(make_incoming(), response="Привет, как дела?")
        assert route_after_response(state) == END

    def test_foreign_script_goes_to_language_correction(self):
        state = make_state(make_incoming(), response="Привет 你好世界 как дела")
        assert route_after_response(state) == "language_correction"


class TestRespondSubgraph:
    """The compiled subgraph keeps every key a node writes: LangGraph drops keys the
    stage schema does not declare, without an error."""

    async def test_meme_request_skips_the_reply_nodes_and_returns_an_empty_reply(self):
        state = make_state(
            make_incoming(raw_text=""), should_respond=True, response_trigger="explicit", meme_request=True,
        )
        result = await build_respond_graph(MagicMock(), MagicMock()).ainvoke(state)
        assert result["response"] == ""
        assert result["worker_status"] == WorkerStatus.SKIPPED
        assert result["response_messages"] == []
        assert result["response_trace"] is None
        assert result["meme_request"] is True
