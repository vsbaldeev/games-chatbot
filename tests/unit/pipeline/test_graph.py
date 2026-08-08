"""
Graph routing-decision tests.

Exercises route_after_router() directly — the pure function that decides
where a message goes after the router node runs. No graph compilation or
LLM calls happen here.
"""

from langgraph.graph import END

from src.pipeline.graph import route_after_router
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
