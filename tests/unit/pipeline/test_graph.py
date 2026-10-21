"""
Graph routing-decision tests.

Exercises the routing functions directly — the pure function that decides
where a message goes after the router node runs. No graph compilation or
LLM calls happen here.
"""

from langgraph.graph import END
import pytest

from src.pipeline.graph import (
    route_after_context,
    route_after_ingester,
    route_after_router,
    route_if_responding,
)
from tests.builders import make_incoming, make_state


class TestRouteAfterRouter:
    def test_routes_to_ingester_when_responding(self):
        state = make_state(make_incoming(raw_text="hello there friend"), should_respond=True)
        assert route_after_router(state) == "ingester"

    def test_routes_to_memory_writer_when_not_responding_and_long(self):
        state = make_state(
            make_incoming(raw_text="a sufficiently long passive message"), should_respond=False
        )
        assert route_after_router(state) == "memory_writer"

    def test_routes_to_end_when_not_responding_and_short(self):
        state = make_state(make_incoming(raw_text="hi"), should_respond=False)
        assert route_after_router(state) == END

    def test_routes_to_ingester_for_media(self):
        state = make_state(make_incoming(media_type="photo", raw_text=None), should_respond=False)
        assert route_after_router(state) == "ingester"


class TestRouteAfterIngester:
    @pytest.mark.parametrize(
        "should_respond, trigger, media_type, expected",
        [
            (False, "explicit", "voice", END),
            (False, "youtube_short", "text", END),
            (True, "youtube_short", "text", "link_check"),
            (True, "social_link", "text", "link_check"),
            (True, "explicit", "voice", "media_check"),
            (True, "explicit", "text", "classify"),
            (True, "insult_check", "text", "overheard"),
        ],
    )
    def test_routes_by_trigger_and_media_type(self, should_respond, trigger, media_type, expected):
        state = make_state(
            make_incoming(media_type=media_type),
            should_respond=should_respond,
            response_trigger=trigger,
        )
        assert route_after_ingester(state) == expected


class TestRouteIfResponding:
    def test_continues_to_next_node_when_responding(self):
        state = make_state(make_incoming(), should_respond=True)
        assert route_if_responding("guard")(state) == "guard"

    def test_ends_when_not_responding(self):
        state = make_state(make_incoming(), should_respond=False)
        assert route_if_responding("guard")(state) == END


class TestRouteAfterContext:
    """Replies that need only personality, or whose source material is already in the
    message, bypass the worker (and its «🔍 Ищу…» notification)."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"is_bot_insult": True},
            {"wind_down": True},
            {"meme_request": True},
            {"group_profile_request": True},
            {"response_trigger": "youtube_short"},
            {"response_trigger": "social_link"},
        ],
        ids=["bot-insult", "wind-down", "meme", "group-profile", "shorts", "social-link"],
    )
    def test_routes_directive_and_link_replies_past_the_worker(self, overrides):
        state = make_state(make_incoming(), should_respond=True, **overrides)
        assert route_after_context(state) == "skip_worker"

    def test_routes_an_ordinary_question_to_the_worker(self):
        state = make_state(make_incoming(), should_respond=True, response_trigger="explicit")
        assert route_after_context(state) == "worker"
