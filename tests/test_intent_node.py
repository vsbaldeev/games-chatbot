"""IntentClassifierNode tests.

Covers two invariants introduced when the classifier was tied to the main
fallback chain:
  1. The classifier uses agent.get_classifier_llm() at call time, not a
     hardwired model — so model fallbacks apply uniformly across the pipeline.
  2. Unexpected raw output is logged as a warning and defaults to "general"
     rather than propagating garbage into route_by_intent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipeline.intent_node import IntentClassifierNode
from tests.builders import make_incoming, make_state

LOGGER_PATH = "src.pipeline.intent_node.logger"


def make_agent(response_text: str) -> MagicMock:
    ai_message = MagicMock()
    ai_message.content = response_text
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=ai_message)
    llm.model_name = "llama-3.3-70b-versatile"
    agent = MagicMock()
    agent.get_classifier_llm.return_value = llm
    return agent


def make_classifier_state(text: str) -> dict:
    incoming = make_incoming(raw_text=text, processed_text=text)
    return make_state(incoming, should_respond=True)


class TestClassifierUsesAgentLlm:
    async def test_get_classifier_llm_called_on_each_invocation(self):
        """The node must fetch the LLM from the agent at call time so that
        model fallbacks (advance_model) take effect without restarting the node."""
        agent = make_agent("games")
        state = make_classifier_state("когда выйдет GTA 6?")

        node = IntentClassifierNode(agent)
        await node(state)

        agent.get_classifier_llm.assert_called_once()

    async def test_uses_llm_returned_by_agent(self):
        """The ainvoke call must go to the LLM returned by get_classifier_llm,
        not to any hardwired model."""
        agent = make_agent("media")
        state = make_classifier_state("посоветуй аниме")

        await IntentClassifierNode(agent)(state)

        agent.get_classifier_llm.return_value.ainvoke.assert_called_once()


class TestValidIntentRouting:
    @pytest.mark.parametrize("llm_output,expected", [
        ("games", "games"),
        ("GAMES", "games"),
        ("media", "media"),
        ("  Media  ", "media"),
        ("general", "general"),
    ])
    async def test_valid_intents_returned_correctly(self, llm_output: str, expected: str):
        agent = make_agent(llm_output)
        state = make_classifier_state("любой текст")

        result = await IntentClassifierNode(agent)(state)

        assert result["intent"] == expected


class TestUnexpectedOutput:
    async def test_unexpected_output_defaults_to_general(self):
        """If the LLM returns something other than games/media/general, the node
        must default to 'general' so the pipeline always has a valid route."""
        agent = make_agent("gaming")
        state = make_classifier_state("вопрос про игры")

        result = await IntentClassifierNode(agent)(state)

        assert result["intent"] == "general"

    async def test_unexpected_output_logged_as_warning(self):
        """Unexpected classifier output must be logged at WARNING so misrouting
        is visible in logs without crashing the pipeline."""
        agent = make_agent("gaming")
        state = make_classifier_state("вопрос про игры")

        with patch(LOGGER_PATH) as mock_logger:
            await IntentClassifierNode(agent)(state)

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Unexpected" in warning_msg

    async def test_empty_output_defaults_to_general(self):
        agent = make_agent("")
        state = make_classifier_state("привет")

        result = await IntentClassifierNode(agent)(state)

        assert result["intent"] == "general"


class TestClassificationFailure:
    async def test_llm_exception_defaults_to_general(self):
        """If the LLM call raises, classification must silently fall back to
        'general' — a routing failure must not crash the pipeline."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=Exception("rate_limit"))
        llm.model_name = "llama-3.3-70b-versatile"
        agent = MagicMock()
        agent.get_classifier_llm.return_value = llm
        state = make_classifier_state("что-то спросили")

        result = await IntentClassifierNode(agent)(state)

        assert result["intent"] == "general"

    async def test_llm_exception_logged_as_warning(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=Exception("rate_limit"))
        llm.model_name = "llama-3.3-70b-versatile"
        agent = MagicMock()
        agent.get_classifier_llm.return_value = llm
        state = make_classifier_state("что-то спросили")

        with patch(LOGGER_PATH) as mock_logger:
            await IntentClassifierNode(agent)(state)

        mock_logger.warning.assert_called_once()
