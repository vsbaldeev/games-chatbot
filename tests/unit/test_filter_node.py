"""MeaninglessFilterNode tests.

Covers:
  - Passthrough when should_respond is already False
  - Text classification: MEANINGLESS, MEANINGFUL, empty/None text
  - Non-text media: explicit trigger passes through, random trigger rejects
  - Reaction emoji correctness, target chat/message, and error handling
  - LLM classify helper: case normalization, fail-open on error
  - REACTION_POOL and FILTER_SYSTEM invariants
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import ReactionTypeEmoji

from src.pipeline.filter_node import FILTER_SYSTEM, REACTION_POOL, MeaninglessFilterNode
from tests.builders import make_incoming, make_state


def close_coroutine(coro):
    """Side-effect for patched asyncio.create_task to avoid unawaited-coroutine warnings."""
    coro.close()


def make_node_with_mock_llm(
    llm_response: str | None = None,
    llm_error: Exception | None = None,
) -> tuple[MeaninglessFilterNode, MagicMock]:
    node = MeaninglessFilterNode()
    mock_llm = MagicMock()
    if llm_error is not None:
        mock_llm.ainvoke = AsyncMock(side_effect=llm_error)
    else:
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content=llm_response or "MEANINGFUL")
        )
    node._MeaninglessFilterNode__llm = mock_llm
    return node, mock_llm


class TestReactionPool:
    def test_reaction_pool_is_not_empty(self):
        assert len(REACTION_POOL) > 0

    def test_all_reactions_are_strings(self):
        assert all(isinstance(emoji, str) for emoji in REACTION_POOL)


class TestFilterSystemPrompt:
    def test_mentions_meaningless_label(self):
        assert "MEANINGLESS" in FILTER_SYSTEM

    def test_mentions_meaningful_label(self):
        assert "MEANINGFUL" in FILTER_SYSTEM

    def test_instructs_single_word_reply(self):
        assert "one word" in FILTER_SYSTEM.lower()

    def test_defaults_to_meaningful_when_uncertain(self):
        assert "unsure" in FILTER_SYSTEM.lower()


class TestPassthroughWhenShouldRespondFalse:
    async def test_returns_empty_dict_when_not_responding(self):
        node, _ = make_node_with_mock_llm()
        state = make_state(make_incoming(), should_respond=False)
        result = await node(state)
        assert result == {}

    async def test_llm_not_called_when_not_responding(self):
        node, mock_llm = make_node_with_mock_llm()
        state = make_state(make_incoming(), should_respond=False)
        await node(state)
        mock_llm.ainvoke.assert_not_called()


class TestTextClassification:
    async def test_meaningless_text_sets_should_respond_false(self):
        node, _ = make_node_with_mock_llm("MEANINGLESS")
        with patch("asyncio.create_task", side_effect=close_coroutine):
            state = make_state(make_incoming(raw_text="ахаха"), should_respond=True)
            result = await node(state)
        assert result == {"should_respond": False}

    async def test_meaningless_text_fires_reaction_task(self):
        node, _ = make_node_with_mock_llm("MEANINGLESS")
        with patch("asyncio.create_task", side_effect=close_coroutine) as mock_create_task:
            state = make_state(make_incoming(raw_text="ахаха"), should_respond=True)
            await node(state)
        mock_create_task.assert_called_once()

    async def test_meaningful_text_sets_should_respond_true(self):
        node, _ = make_node_with_mock_llm("MEANINGFUL")
        state = make_state(make_incoming(raw_text="расскажи про GTA 6"), should_respond=True)
        result = await node(state)
        assert result == {"should_respond": True}

    async def test_meaningful_text_does_not_fire_reaction(self):
        node, _ = make_node_with_mock_llm("MEANINGFUL")
        with patch("asyncio.create_task", side_effect=close_coroutine) as mock_create_task:
            state = make_state(make_incoming(raw_text="расскажи про GTA 6"), should_respond=True)
            await node(state)
        mock_create_task.assert_not_called()

    async def test_empty_text_rejects_without_llm_call(self):
        node, mock_llm = make_node_with_mock_llm()
        state = make_state(make_incoming(raw_text="   "), should_respond=True)
        result = await node(state)
        assert result == {"should_respond": False}
        mock_llm.ainvoke.assert_not_called()

    async def test_none_text_rejects_without_llm_call(self):
        node, mock_llm = make_node_with_mock_llm()
        state = make_state(make_incoming(raw_text=None), should_respond=True)
        result = await node(state)
        assert result == {"should_respond": False}
        mock_llm.ainvoke.assert_not_called()


class TestMediaMessages:
    async def test_no_transcription_explicit_gets_canned_failure_reply(self):
        """An addressed voice note with no transcript gets an honest canned
        reply instead of a pass-through that would hallucinate a reaction."""
        from src.pipeline.filter_node import TRANSCRIPTION_FAILED_REPLIES

        node, _ = make_node_with_mock_llm()
        state = make_state(
            make_incoming(media_type="voice", processed_text=None),
            should_respond=True,
            response_trigger="explicit",
        )
        result = await node(state)
        assert result["should_respond"] is False
        assert result["response"] in TRANSCRIPTION_FAILED_REPLIES

    async def test_no_transcription_explicit_does_not_fire_reaction(self):
        node, _ = make_node_with_mock_llm()
        with patch("asyncio.create_task", side_effect=close_coroutine) as mock_create_task:
            state = make_state(
                make_incoming(media_type="voice", processed_text=None),
                should_respond=True,
                response_trigger="explicit",
            )
            await node(state)
        mock_create_task.assert_not_called()

    async def test_no_transcription_random_sets_should_respond_false(self):
        node, _ = make_node_with_mock_llm()
        with patch("asyncio.create_task", side_effect=close_coroutine):
            state = make_state(
                make_incoming(media_type="voice", processed_text=None),
                should_respond=True,
                response_trigger="random",
            )
            result = await node(state)
        assert result == {"should_respond": False}

    async def test_no_transcription_random_fires_reaction(self):
        node, _ = make_node_with_mock_llm()
        with patch("asyncio.create_task", side_effect=close_coroutine) as mock_create_task:
            state = make_state(
                make_incoming(media_type="voice", processed_text=None),
                should_respond=True,
                response_trigger="random",
            )
            await node(state)
        mock_create_task.assert_called_once()

    async def test_media_with_transcription_passes_through(self):
        node, _ = make_node_with_mock_llm()
        state = make_state(
            make_incoming(media_type="voice", processed_text="расскажи про игру"),
            should_respond=True,
        )
        result = await node(state)
        assert result == {}

    async def test_whitespace_only_transcription_treated_as_missing(self):
        node, _ = make_node_with_mock_llm()
        with patch("asyncio.create_task", side_effect=close_coroutine):
            state = make_state(
                make_incoming(media_type="voice", processed_text="   "),
                should_respond=True,
                response_trigger="random",
            )
            result = await node(state)
        assert result == {"should_respond": False}


class TestClassify:
    async def test_meaningless_llm_response_returns_meaningless(self):
        node, _ = make_node_with_mock_llm("MEANINGLESS")
        result = await node._MeaninglessFilterNode__classify("лол", FILTER_SYSTEM)
        assert result == "MEANINGLESS"

    async def test_meaningful_llm_response_returns_meaningful(self):
        node, _ = make_node_with_mock_llm("MEANINGFUL")
        result = await node._MeaninglessFilterNode__classify("как дела?", FILTER_SYSTEM)
        assert result == "MEANINGFUL"

    async def test_llm_error_fails_open_as_meaningful(self):
        node, _ = make_node_with_mock_llm(llm_error=Exception("API unavailable"))
        result = await node._MeaninglessFilterNode__classify("лол", FILTER_SYSTEM)
        assert result == "MEANINGFUL"

    async def test_lowercase_response_parsed_as_meaningless(self):
        node, _ = make_node_with_mock_llm("meaningless")
        result = await node._MeaninglessFilterNode__classify("хаха", FILTER_SYSTEM)
        assert result == "MEANINGLESS"

    async def test_response_with_trailing_text_still_parsed(self):
        node, _ = make_node_with_mock_llm("MEANINGLESS - just laughter")
        result = await node._MeaninglessFilterNode__classify("хаха", FILTER_SYSTEM)
        assert result == "MEANINGLESS"

    async def test_unrecognized_response_defaults_to_meaningful(self):
        node, _ = make_node_with_mock_llm("UNKNOWN")
        result = await node._MeaninglessFilterNode__classify("что-то", FILTER_SYSTEM)
        assert result == "MEANINGFUL"


class TestSendReaction:
    async def test_sends_reaction_from_pool(self):
        node = MeaninglessFilterNode()
        state = make_state(make_incoming())
        mock_bot = AsyncMock()
        state["context_types"].bot = mock_bot

        await node._MeaninglessFilterNode__send_reaction(state)

        mock_bot.set_message_reaction.assert_called_once()
        sent_reaction = mock_bot.set_message_reaction.call_args.kwargs["reaction"][0]
        assert isinstance(sent_reaction, ReactionTypeEmoji)
        assert sent_reaction.emoji in REACTION_POOL

    async def test_sends_to_correct_chat_and_message(self):
        node = MeaninglessFilterNode()
        state = make_state(make_incoming(chat_id=9999, message_id=777))
        mock_bot = AsyncMock()
        state["context_types"].bot = mock_bot

        await node._MeaninglessFilterNode__send_reaction(state)

        call_kwargs = mock_bot.set_message_reaction.call_args.kwargs
        assert call_kwargs["chat_id"] == 9999
        assert call_kwargs["message_id"] == 777

    async def test_bot_error_is_swallowed(self):
        node = MeaninglessFilterNode()
        state = make_state(make_incoming())
        mock_bot = AsyncMock()
        mock_bot.set_message_reaction.side_effect = Exception("Telegram API down")
        state["context_types"].bot = mock_bot

        await node._MeaninglessFilterNode__send_reaction(state)
