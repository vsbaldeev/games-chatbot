"""OverheardNode tests: bot-word mentions are acted on only after a second opinion."""

from unittest.mock import patch

from src.pipeline.state import Verdict
from tests.builders import make_incoming, make_overheard_node, make_state

DROPPED = {"should_respond": False, "drop_reason": "overheard_dropped"}
EXTRACT_TARGET = "src.pipeline.ingest.filter.overheard.extract_passive_facts"


def make_overheard_state(text: str = "этот бот вообще тупой") -> dict:
    return make_state(
        make_incoming(raw_text=text), should_respond=True, response_trigger="insult_check",
    )


class TestOverheardNode:
    async def test_confirmed_insult_sets_bot_insult_verdict(self):
        node = make_overheard_node(cheap_response="BOT_INSULT", confirm_response="BOT_INSULT")
        with patch(EXTRACT_TARGET):
            result = await node(make_overheard_state())
        assert result == {"filter_verdict": Verdict.BOT_INSULT}

    async def test_unconfirmed_insult_is_dropped(self):
        node = make_overheard_node(cheap_response="BOT_INSULT", confirm_response="OTHER")
        with patch(EXTRACT_TARGET):
            assert await node(make_overheard_state()) == DROPPED

    async def test_confirmation_error_is_dropped(self):
        node = make_overheard_node(cheap_response="BOT_INSULT", confirm_response=RuntimeError("down"))
        with patch(EXTRACT_TARGET):
            assert await node(make_overheard_state()) == DROPPED

    async def test_non_insult_is_dropped_without_confirmation(self):
        node = make_overheard_node(cheap_response="OTHER", confirm_response="BOT_INSULT")
        with patch(EXTRACT_TARGET):
            assert await node(make_overheard_state()) == DROPPED
        node.confirm_llm_mock.ainvoke.assert_not_called()

    async def test_dropped_text_schedules_passive_extraction(self):
        node = make_overheard_node(cheap_response="OTHER", confirm_response="OTHER")
        state = make_overheard_state()
        with patch(EXTRACT_TARGET) as extract:
            await node(state)
        extract.assert_called_once_with(state["incoming"])
