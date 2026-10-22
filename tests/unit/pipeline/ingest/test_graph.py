"""Ingest subgraph routing-decision tests: which filter node handles a message.
"""

from langgraph.graph import END
import pytest

from src.pipeline.ingest.graph import route_after_ingester, route_if_responding
from tests.builders import make_incoming, make_state


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
