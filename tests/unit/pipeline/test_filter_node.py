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

from src import config
from src.config.prompts import GROUP_PROFILE_COOLDOWN_REPLIES
from src.pipeline.state import ContentKind
from src.pipeline import engagement_gate
from src.pipeline.filter_node import (
    FILTER_SYSTEM,
    GROUP_PROFILE_MARKER_RE,
    REACTION_POOL,
    MeaninglessFilterNode,
    looks_like_request,
)
from src.pipeline.router import MessageRouter
from src.utils.ttl_gate import TtlGate
from tests.builders import make_incoming, make_state, make_telegram_message


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
        assert result == {
            "should_respond": False,
            "drop_reason": "meaningless",
            "filter_verdict": "MEANINGLESS",
            "engagement_tier": 1,
        }

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
        assert result == {
            "should_respond": True,
            "filter_verdict": "MEANINGFUL",
            "engagement_tier": 1,
        }

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
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE

    async def test_no_vision_explicit_photo_is_a_media_failure_notice(self):
        from src.pipeline.filter_node import VISION_FAILED_REPLIES

        node, _ = make_node_with_mock_llm()
        state = make_state(
            make_incoming(media_type="photo", processed_text=None),
            should_respond=True,
            response_trigger="explicit",
        )
        result = await node(state)
        assert result["response"] in VISION_FAILED_REPLIES
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE

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

    async def test_media_with_transcription_charges_budget_as_meaningful(self):
        node, mock_llm = make_node_with_mock_llm()
        state = make_state(
            make_incoming(media_type="voice", processed_text="расскажи про игру"),
            should_respond=True,
            response_trigger="explicit",
        )
        result = await node(state)
        assert result == {
            "should_respond": True,
            "filter_verdict": "MEANINGFUL",
            "engagement_tier": 1,
        }
        mock_llm.ainvoke.assert_not_called()

    async def test_whitespace_only_transcription_treated_as_missing(self):
        node, _ = make_node_with_mock_llm()
        state = make_state(
            make_incoming(media_type="voice", processed_text="   "),
            should_respond=True,
            response_trigger="explicit",
        )
        result = await node(state)
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE


def make_filter_node() -> MeaninglessFilterNode:
    with patch("src.pipeline.filter_node.make_filter_llm"):
        return MeaninglessFilterNode()


class TestNotAddressedVerdict:
    """A NOT_ADDRESSED verdict silences the bot, but only for un-mentioned
    replies to a broadcast. 2026-08-18 incident: members replied to a
    group-wide ratings post to argue with each other and got answered.
    """

    async def test_broadcast_reply_is_dropped_silently(self):
        node, _ = make_node_with_mock_llm("NOT_ADDRESSED")
        # Under SUBSTANTIVE_WORD_COUNT and not a question/imperative, so the
        # looks_like_request floor (Finding 1) does not itself downgrade it —
        # this test is about the silent-drop shape, not the floor.
        text = "да ладно тебе, отстань уже"
        state = make_state(
            make_incoming(raw_text=text),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch(
            "src.pipeline.filter_node.asyncio.create_task", side_effect=close_coroutine
        ) as mock_task, patch("src.pipeline.filter_node.extract_and_save", new=AsyncMock()):
            result = await node(state)
        assert result["should_respond"] is False
        assert result["drop_reason"] == "broadcast_not_addressed"
        # Only the passive-extraction task fires (the text is long enough) —
        # no emoji reaction, no reply.
        mock_task.assert_called_once()

    async def test_not_addressed_charges_no_attention_budget(self):
        """The bot was never addressed — charging the user would wind them
        down for talking to their friends."""
        node, _ = make_node_with_mock_llm("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="Гениально)"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch(
            "src.pipeline.filter_node.engagement_gate.register_signal",
            new_callable=AsyncMock,
        ) as mock_register, patch("src.pipeline.filter_node.asyncio.create_task"):
            await node(state)
        mock_register.assert_not_called()

    async def test_not_addressed_outside_a_broadcast_reply_is_downgraded(self):
        """Ordinary conversation can never be silenced by this verdict."""
        node, _ = make_node_with_mock_llm("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="ну он такое себе"),
            should_respond=True, response_trigger="explicit",
        )
        with patch(
            "src.pipeline.filter_node.engagement_gate.register_signal",
            new_callable=AsyncMock, return_value=engagement_gate.FULL_TIER,
        ):
            result = await node(state)
        assert result["should_respond"] is True
        assert result["filter_verdict"] == "MEANINGFUL"

    async def test_question_on_a_broadcast_reply_still_answers(self):
        """The classifier's own MEANINGFUL verdict is untouched by the gate."""
        node, _ = make_node_with_mock_llm("MEANINGFUL")
        state = make_state(
            make_incoming(raw_text="а почему мне 2 из 10?"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch(
            "src.pipeline.filter_node.engagement_gate.register_signal",
            new_callable=AsyncMock, return_value=engagement_gate.FULL_TIER,
        ):
            result = await node(state)
        assert result["should_respond"] is True

    async def test_not_addressed_question_on_broadcast_reply_is_downgraded(self):
        """Finding 1: a NOT_ADDRESSED misfire on a real question is never
        silenced — looks_like_request is the same deterministic floor
        MEANINGLESS/BANTER already get, applied to the one verdict that
        produces total silence."""
        node, _ = make_node_with_mock_llm("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="а почему мне 2 из 10?"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch(
            "src.pipeline.filter_node.engagement_gate.register_signal",
            new_callable=AsyncMock, return_value=engagement_gate.FULL_TIER,
        ):
            result = await node(state)
        assert result["should_respond"] is True
        assert result["filter_verdict"] == "MEANINGFUL"

    async def test_not_addressed_imperative_request_on_broadcast_reply_is_downgraded(self):
        """The floor covers imperative requests too, not just question marks."""
        node, _ = make_node_with_mock_llm("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="объясни за что мне такая оценка"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch(
            "src.pipeline.filter_node.engagement_gate.register_signal",
            new_callable=AsyncMock, return_value=engagement_gate.FULL_TIER,
        ):
            result = await node(state)
        assert result["should_respond"] is True
        assert result["filter_verdict"] == "MEANINGFUL"

    async def test_classify_recognises_the_label(self):
        node, _ = make_node_with_mock_llm("not_addressed")
        verdict = await node._MeaninglessFilterNode__classify("текст", "system")
        assert verdict == "NOT_ADDRESSED"


class TestNotAddressedPassiveExtraction:
    """Finding 2: a message silently dropped as NOT_ADDRESSED must still run
    passive fact extraction, same as the overheard-drop path — otherwise
    substantive self-description (the incident's own example: a member
    describing their own effort while venting about the bot's post) is lost.
    """

    async def test_long_text_schedules_extraction_with_the_sender_identity(self):
        node, _ = make_node_with_mock_llm("NOT_ADDRESSED")
        # At MIN_PASSIVE_LENGTH but at/under SUBSTANTIVE_WORD_COUNT and not a
        # question/imperative, so it stays NOT_ADDRESSED (Finding 1's floor
        # does not fire) and reaches the extraction call under test.
        text = "он мне вообще не пишет никогда"
        state = make_state(
            make_incoming(raw_text=text, chat_id=4242, user_id=77, username="vasya"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch(
            "src.pipeline.filter_node.asyncio.create_task", side_effect=close_coroutine
        ), patch("src.pipeline.filter_node.extract_and_save", new=AsyncMock()) as mock_extract:
            await node(state)
        mock_extract.assert_called_once_with(
            chat_id=4242, user_id=77, username="vasya", user_message=text,
        )

    async def test_short_text_does_not_schedule_extraction(self):
        node, _ = make_node_with_mock_llm("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="Гениально)"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch("src.pipeline.filter_node.asyncio.create_task") as mock_task:
            await node(state)
        mock_task.assert_not_called()


class TestRouterToFilterNodeWiring:
    """Finding 3: ``broadcast_reply`` is produced by ``MessageRouter.__decide``
    and consumed by ``MeaninglessFilterNode`` purely through an untyped
    ``NotRequired`` state key — a typo on either side would fail open
    silently (the bot would answer as if this whole gate did not exist) and
    every other test would stay green, because each layer's tests mock the
    other layer's output instead of actually connecting them. This test
    flows the router's real return value into the filter node's real input,
    with no hand-built intermediate state.
    """

    async def test_broadcast_reply_flows_from_router_into_a_silent_drop(self):
        router = MessageRouter(bot_username=config.BOT_USERNAME, bot_id=config.BOT_ID)
        text = "Гениально)"
        telegram_message = make_telegram_message(reply_to_user_id=config.BOT_ID, text=text)
        incoming = make_incoming(raw_text=text, telegram_message=telegram_message)

        with patch(
            "src.pipeline.router.unified_messages.get_by_id",
            new_callable=AsyncMock, return_value={"is_broadcast": True},
        ):
            router_update = await router._MessageRouter__decide(incoming, telegram_message)

        assert router_update == {
            "should_respond": True,
            "response_trigger": "explicit",
            "broadcast_reply": True,
        }

        state = make_state(incoming, **router_update)
        node, _ = make_node_with_mock_llm("NOT_ADDRESSED")
        with patch("src.pipeline.filter_node.asyncio.create_task", side_effect=close_coroutine):
            result = await node(state)

        assert result["should_respond"] is False
        assert result["drop_reason"] == "broadcast_not_addressed"
