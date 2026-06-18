"""
Part 3 — autonomous-humor opportunity gate.

The gate is pure logic over per-chat in-memory counters: it decides whether a
message is even worth handing to the comedian, keeping the LLM off the hot path.
These tests pin the cadence rules (min gap, cooldown, joke-worthiness,
probability) and the graph routing into the pass-through stub.
"""

from unittest.mock import patch

import pytest

from src.pipeline import humor_gate
from src.pipeline.graph import route_after_router

CHAT_ID = 555


def make_msg(text: str = "a long enough message to consider", *,
             media_type: str = "text", forwarded: bool = False) -> dict:
    return {
        "chat_id": CHAT_ID,
        "media_type": media_type,
        "raw_text": text,
        "is_forwarded": forwarded,
    }


@pytest.fixture(autouse=True)
def clear_gate_state():
    humor_gate.messages_since_joke.clear()
    humor_gate.last_joke_time.clear()
    yield
    humor_gate.messages_since_joke.clear()
    humor_gate.last_joke_time.clear()


def make_eligible():
    """Put a chat into a state where every non-random condition is satisfied."""
    humor_gate.messages_since_joke[CHAT_ID] = humor_gate.MIN_MESSAGES_SINCE_JOKE
    humor_gate.last_joke_time.pop(CHAT_ID, None)


class TestObserveAndMarks:
    def test_observe_increments_counter(self):
        humor_gate.observe(CHAT_ID)
        humor_gate.observe(CHAT_ID)
        assert humor_gate.messages_since_joke[CHAT_ID] == 2

    def test_mark_considered_resets_counter(self):
        humor_gate.messages_since_joke[CHAT_ID] = 99
        humor_gate.mark_considered(CHAT_ID)
        assert humor_gate.messages_since_joke[CHAT_ID] == 0

    def test_mark_joke_sent_resets_counter_and_stamps_cooldown(self):
        humor_gate.messages_since_joke[CHAT_ID] = 99
        with patch("src.pipeline.humor_gate.time.time", return_value=1_000.0):
            humor_gate.mark_joke_sent(CHAT_ID)
        assert humor_gate.messages_since_joke[CHAT_ID] == 0
        assert humor_gate.last_joke_time[CHAT_ID] == 1_000.0


class TestShouldConsider:
    def test_fires_when_all_conditions_met(self):
        make_eligible()
        with patch("src.pipeline.humor_gate.random.random", return_value=0.0), \
             patch("src.pipeline.humor_gate.time.time", return_value=10_000_000.0):
            assert humor_gate.should_consider(CHAT_ID, make_msg()) is True

    def test_blocked_below_min_gap(self):
        humor_gate.messages_since_joke[CHAT_ID] = humor_gate.MIN_MESSAGES_SINCE_JOKE - 1
        with patch("src.pipeline.humor_gate.random.random", return_value=0.0), \
             patch("src.pipeline.humor_gate.time.time", return_value=10_000_000.0):
            assert humor_gate.should_consider(CHAT_ID, make_msg()) is False

    def test_blocked_during_cooldown(self):
        humor_gate.messages_since_joke[CHAT_ID] = humor_gate.MIN_MESSAGES_SINCE_JOKE
        humor_gate.last_joke_time[CHAT_ID] = 1_000.0
        with patch("src.pipeline.humor_gate.random.random", return_value=0.0), \
             patch("src.pipeline.humor_gate.time.time", return_value=1_000.0 + humor_gate.COOLDOWN_SECONDS - 1):
            assert humor_gate.should_consider(CHAT_ID, make_msg()) is False

    def test_blocked_by_probability_roll(self):
        make_eligible()
        with patch("src.pipeline.humor_gate.random.random", return_value=1.0), \
             patch("src.pipeline.humor_gate.time.time", return_value=10_000_000.0):
            assert humor_gate.should_consider(CHAT_ID, make_msg()) is False

    def test_blocked_for_short_text(self):
        make_eligible()
        with patch("src.pipeline.humor_gate.random.random", return_value=0.0), \
             patch("src.pipeline.humor_gate.time.time", return_value=10_000_000.0):
            assert humor_gate.should_consider(CHAT_ID, make_msg("коротко")) is False

    def test_blocked_for_forwarded(self):
        make_eligible()
        with patch("src.pipeline.humor_gate.random.random", return_value=0.0), \
             patch("src.pipeline.humor_gate.time.time", return_value=10_000_000.0):
            assert humor_gate.should_consider(CHAT_ID, make_msg(forwarded=True)) is False

    def test_blocked_for_non_text(self):
        make_eligible()
        with patch("src.pipeline.humor_gate.random.random", return_value=0.0), \
             patch("src.pipeline.humor_gate.time.time", return_value=10_000_000.0):
            assert humor_gate.should_consider(CHAT_ID, make_msg(media_type="photo")) is False

    def test_rapid_burst_never_trips(self):
        """After a consideration, a fresh burst stays below the gap and is silent."""
        humor_gate.mark_considered(CHAT_ID)
        fired = False
        with patch("src.pipeline.humor_gate.random.random", return_value=0.0), \
             patch("src.pipeline.humor_gate.time.time", return_value=10_000_000.0):
            for _ in range(humor_gate.MIN_MESSAGES_SINCE_JOKE - 1):
                humor_gate.observe(CHAT_ID)
                fired = fired or humor_gate.should_consider(CHAT_ID, make_msg())
        assert fired is False


class TestRouteAfterRouter:
    def make_state(self, *, should_respond: bool, text: str, forwarded: bool = False):
        return {
            "incoming": make_msg(text, forwarded=forwarded),
            "should_respond": should_respond,
        }

    def test_routes_to_ingester_when_responding(self):
        state = self.make_state(should_respond=True, text="hello there friend")
        assert route_after_router(state) == "ingester"

    def test_routes_to_humor_when_gate_fires(self):
        state = self.make_state(should_respond=False, text="a juicy long opener message")
        with patch("src.pipeline.humor_gate.should_consider", return_value=True):
            assert route_after_router(state) == "humor"

    def test_routes_to_memory_writer_when_gate_false_and_long(self):
        state = self.make_state(should_respond=False, text="a sufficiently long passive message")
        with patch("src.pipeline.humor_gate.should_consider", return_value=False):
            assert route_after_router(state) == "memory_writer"

    def test_routes_to_end_when_gate_false_and_short(self):
        from langgraph.graph import END
        state = self.make_state(should_respond=False, text="hi")
        with patch("src.pipeline.humor_gate.should_consider", return_value=False):
            assert route_after_router(state) == END
