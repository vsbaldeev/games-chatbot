"""
Graph routing-decision tests.

Exercises the routing functions directly — the pure function that decides
where a message goes after the router node runs. No graph compilation or
LLM calls happen here.
"""

from langgraph.graph import END, START, StateGraph
import pytest

from src.pipeline.graph import route_after_respond, route_after_router
from src.pipeline.state import BotState, RespondState
from tests.builders import make_incoming, make_state


class TestRouteAfterRouter:
    def test_routes_to_ingester_when_responding(self):
        state = make_state(make_incoming(raw_text="hello there friend"), should_respond=True)
        assert route_after_router(state) == "ingest"

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
        assert route_after_router(state) == "ingest"




class TestRouteAfterRespond:
    def test_ends_a_blocked_run_without_storing_memories(self):
        state = make_state(make_incoming(), should_respond=True, blocked=True)
        assert route_after_respond(state) == END

    def test_hands_a_delivered_reply_to_the_memory_writer(self):
        state = make_state(make_incoming(), should_respond=True)
        assert route_after_respond(state) == "memory_writer"


class TestStreamedSubgraphStates:
    """run_pipeline streams with ``subgraphs=True`` so a failure inside the respond
    stage still leaves the worker's search notification in the merged state."""

    async def test_inner_states_survive_a_failure_in_a_later_subgraph_node(self):
        async def worker(state):
            return {"search_notification_msg": "notice"}

        async def response(state):
            raise RuntimeError("response model exhausted")

        respond = StateGraph(RespondState)
        respond.add_node("worker", worker)
        respond.add_node("response", response)
        respond.add_edge(START, "worker")
        respond.add_edge("worker", "response")
        parent = StateGraph(BotState)
        parent.add_node("respond", respond.compile())
        parent.add_edge(START, "respond")

        merged: dict = dict(make_state(make_incoming(), should_respond=True))
        with pytest.raises(RuntimeError):
            async for _namespace, values in parent.compile().astream(
                merged, stream_mode="values", subgraphs=True
            ):
                merged = {**merged, **values}
        assert merged["search_notification_msg"] == "notice"
