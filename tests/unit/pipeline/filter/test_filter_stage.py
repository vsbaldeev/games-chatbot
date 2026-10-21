"""Filter stage tests: the four filter nodes wired by ``add_filter_stage``.

Runs the stage as a small compiled graph (``START → … → END``), so these tests
cover the real routing between link_check, media_check, classify and budget.

Covers:
  - Passthrough when should_respond is already False
  - Text classification: MEANINGLESS, MEANINGFUL, empty/None text
  - Non-text media: transcribed media is charged, empty media gets a canned reply
  - Shorts / social links bypass classification
  - NOT_ADDRESSED handling and passive fact extraction on silent drops
  - Router → filter wiring of ``broadcast_reply``

Patching ``asyncio.create_task`` globally breaks LangGraph's own scheduling, so
the fire-and-forget side effects (``send_reaction``, ``extract_and_save``) are
patched at their source instead.
"""

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.graph import END, START, StateGraph

from src import config
from src.pipeline import engagement_gate
from src.pipeline.filter.classify import ClassifyNode
from src.pipeline.graph import add_filter_stage
from src.pipeline.router import MessageRouter
from src.pipeline.state import BotState, ContentKind
from tests.builders import make_incoming, make_state, make_telegram_message


def make_classify_node(llm_response: str | None = None) -> tuple[ClassifyNode, MagicMock]:
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=llm_response or "MEANINGFUL"))
    return ClassifyNode(llm=mock_llm, confirm_llm=mock_llm), mock_llm


async def run_filter_stage(state: dict, classify_node: ClassifyNode) -> dict:
    """Run the filter stage on ``state`` and return the merged node updates."""
    graph = StateGraph(BotState)
    add_filter_stage(graph, source=START, on_pass=END, classify_node=classify_node)
    merged: dict = {}
    async for chunk in graph.compile().astream(state, stream_mode="updates"):
        for update in chunk.values():
            merged.update(update or {})
    await asyncio.sleep(0)  # let fire-and-forget tasks created by the nodes run
    return merged


@contextmanager
def patched_side_effects():
    """Patch the filter's fire-and-forget calls; yields ``(send_reaction, extract_and_save)``."""
    with patch("src.pipeline.filter.budget.send_reaction", new=AsyncMock()) as reaction, patch(
        "src.pipeline.filter.classify.extract_and_save", new=AsyncMock()
    ) as extract:
        yield reaction, extract


def patch_register_signal(tier: int = engagement_gate.FULL_TIER):
    return patch(
        "src.pipeline.filter.budget.engagement_gate.register_signal",
        new_callable=AsyncMock, return_value=tier,
    )


class TestPassthroughWhenShouldRespondFalse:
    async def test_returns_empty_dict_when_not_responding(self):
        node, _ = make_classify_node()
        state = make_state(make_incoming(), should_respond=False)
        assert await run_filter_stage(state, node) == {}

    async def test_llm_not_called_when_not_responding(self):
        node, mock_llm = make_classify_node()
        state = make_state(make_incoming(), should_respond=False)
        await run_filter_stage(state, node)
        mock_llm.ainvoke.assert_not_called()


class TestTextClassification:
    async def test_meaningless_text_sets_should_respond_false(self):
        node, _ = make_classify_node("MEANINGLESS")
        with patched_side_effects():
            state = make_state(make_incoming(raw_text="ахаха"), should_respond=True)
            result = await run_filter_stage(state, node)
        assert result == {
            "should_respond": False,
            "drop_reason": "meaningless",
            "filter_verdict": "MEANINGLESS",
            "engagement_tier": 1,
        }

    async def test_meaningless_text_fires_reaction(self):
        node, _ = make_classify_node("MEANINGLESS")
        with patched_side_effects() as (reaction, _):
            state = make_state(make_incoming(raw_text="ахаха"), should_respond=True)
            await run_filter_stage(state, node)
        reaction.assert_called_once()

    async def test_meaningful_text_sets_should_respond_true(self):
        node, _ = make_classify_node("MEANINGFUL")
        state = make_state(make_incoming(raw_text="расскажи про GTA 6"), should_respond=True)
        result = await run_filter_stage(state, node)
        assert result == {
            "should_respond": True,
            "filter_verdict": "MEANINGFUL",
            "engagement_tier": 1,
        }

    async def test_meaningful_text_does_not_fire_reaction(self):
        node, _ = make_classify_node("MEANINGFUL")
        with patched_side_effects() as (reaction, _):
            state = make_state(make_incoming(raw_text="расскажи про GTA 6"), should_respond=True)
            await run_filter_stage(state, node)
        reaction.assert_not_called()

    async def test_mentioned_question_survives_a_meaningless_verdict(self):
        """A mentioned question no longer drops to an emoji."""
        node, _ = make_classify_node("MEANINGLESS")
        with patched_side_effects() as (reaction, _):
            state = make_state(
                make_incoming(raw_text="@zhora_bot о чем я говорю"), should_respond=True
            )
            result = await run_filter_stage(state, node)
        assert result["should_respond"] is True
        assert result["filter_verdict"] == "MEANINGFUL"
        reaction.assert_not_called()

    async def test_empty_text_rejects_without_llm_call(self):
        node, mock_llm = make_classify_node()
        state = make_state(make_incoming(raw_text="   "), should_respond=True)
        assert await run_filter_stage(state, node) == {"should_respond": False}
        mock_llm.ainvoke.assert_not_called()

    async def test_none_text_rejects_without_llm_call(self):
        node, mock_llm = make_classify_node()
        state = make_state(make_incoming(raw_text=None), should_respond=True)
        assert await run_filter_stage(state, node) == {"should_respond": False}
        mock_llm.ainvoke.assert_not_called()


class TestMediaMessages:
    async def test_no_transcription_gets_canned_reply_and_no_reaction(self):
        node, _ = make_classify_node()
        with patched_side_effects() as (reaction, _):
            state = make_state(
                make_incoming(media_type="voice", processed_text=None),
                should_respond=True,
                response_trigger="explicit",
            )
            result = await run_filter_stage(state, node)
        assert result["should_respond"] is False
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE
        reaction.assert_not_called()

    async def test_media_with_transcription_charges_budget_as_meaningful(self):
        node, mock_llm = make_classify_node()
        state = make_state(
            make_incoming(media_type="voice", processed_text="расскажи про игру"),
            should_respond=True,
            response_trigger="explicit",
        )
        result = await run_filter_stage(state, node)
        assert result == {
            "should_respond": True,
            "filter_verdict": "MEANINGFUL",
            "engagement_tier": 1,
        }
        mock_llm.ainvoke.assert_not_called()


class TestLinkTriggers:
    async def test_shorts_with_content_passes_without_classification_or_budget(self):
        node, mock_llm = make_classify_node()
        state = make_state(
            make_incoming(raw_text="https://www.youtube.com/shorts/abc"),
            should_respond=True, response_trigger="youtube_short",
            youtube_short_content="[YouTube Shorts]\nsome content",
        )
        with patch_register_signal() as mock_register:
            result = await run_filter_stage(state, node)
        assert result == {"filter_verdict": "SHORTS"}
        mock_llm.ainvoke.assert_not_called()
        mock_register.assert_not_called()

    async def test_social_link_without_content_ends_the_run(self):
        node, _ = make_classify_node()
        telegram_message = MagicMock(text="https://www.reddit.com/r/x/comments/abc/y/", caption=None)
        telegram_message.reply_to_message = None
        incoming = make_incoming(
            raw_text="https://www.reddit.com/r/x/comments/abc/y/", telegram_message=telegram_message,
        )
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link", social_link_content=None,
        )
        result = await run_filter_stage(state, node)
        assert result == {"should_respond": False, "drop_reason": "social_link_failed"}


class TestNotAddressedVerdict:
    """A NOT_ADDRESSED verdict silences the bot, but only for un-mentioned
    replies to a broadcast. 2026-08-18 incident: members replied to a
    group-wide ratings post to argue with each other and got answered.
    """

    async def test_broadcast_reply_is_dropped_silently(self):
        node, _ = make_classify_node("NOT_ADDRESSED")
        # Under SUBSTANTIVE_WORD_COUNT and not a question/imperative, so the
        # looks_like_request floor does not itself downgrade it — this test is
        # about the silent-drop shape, not the floor.
        text = "да ладно тебе, отстань уже"
        state = make_state(
            make_incoming(raw_text=text),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patched_side_effects() as (reaction, extract):
            result = await run_filter_stage(state, node)
        assert result["should_respond"] is False
        assert result["drop_reason"] == "broadcast_not_addressed"
        # Only the passive extraction fires (the text is long enough) — no
        # emoji reaction, no reply.
        extract.assert_called_once()
        reaction.assert_not_called()

    async def test_not_addressed_charges_no_attention_budget(self):
        """The bot was never addressed — charging the user would wind them
        down for talking to their friends."""
        node, _ = make_classify_node("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="Гениально)"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patched_side_effects(), patch_register_signal() as mock_register:
            await run_filter_stage(state, node)
        mock_register.assert_not_called()

    async def test_not_addressed_outside_a_broadcast_reply_is_downgraded(self):
        """Ordinary conversation can never be silenced by this verdict."""
        node, _ = make_classify_node("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="ну он такое себе"),
            should_respond=True, response_trigger="explicit",
        )
        with patch_register_signal():
            result = await run_filter_stage(state, node)
        assert result["should_respond"] is True
        assert result["filter_verdict"] == "MEANINGFUL"

    async def test_question_on_a_broadcast_reply_still_answers(self):
        """The classifier's own MEANINGFUL verdict is untouched by the gate."""
        node, _ = make_classify_node("MEANINGFUL")
        state = make_state(
            make_incoming(raw_text="а почему мне 2 из 10?"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch_register_signal():
            result = await run_filter_stage(state, node)
        assert result["should_respond"] is True

    async def test_not_addressed_question_on_broadcast_reply_is_downgraded(self):
        """A NOT_ADDRESSED misfire on a real question is never silenced —
        looks_like_request is the same deterministic floor MEANINGLESS/BANTER
        already get, applied to the one verdict that produces total silence."""
        node, _ = make_classify_node("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="а почему мне 2 из 10?"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch_register_signal():
            result = await run_filter_stage(state, node)
        assert result["should_respond"] is True
        assert result["filter_verdict"] == "MEANINGFUL"

    async def test_not_addressed_imperative_request_on_broadcast_reply_is_downgraded(self):
        """The floor covers imperative requests too, not just question marks."""
        node, _ = make_classify_node("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="объясни за что мне такая оценка"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patch_register_signal():
            result = await run_filter_stage(state, node)
        assert result["should_respond"] is True
        assert result["filter_verdict"] == "MEANINGFUL"


class TestNotAddressedPassiveExtraction:
    """A message silently dropped as NOT_ADDRESSED must still run passive fact
    extraction, same as the overheard-drop path — otherwise substantive
    self-description (a member describing their own effort while venting about
    the bot's post) is lost.
    """

    async def test_long_text_schedules_extraction_with_the_sender_identity(self):
        node, _ = make_classify_node("NOT_ADDRESSED")
        # At MIN_PASSIVE_LENGTH but at/under SUBSTANTIVE_WORD_COUNT and not a
        # question/imperative, so it stays NOT_ADDRESSED and reaches the
        # extraction call under test.
        text = "он мне вообще не пишет никогда"
        state = make_state(
            make_incoming(raw_text=text, chat_id=4242, user_id=77, username="vasya"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patched_side_effects() as (_, extract):
            await run_filter_stage(state, node)
        extract.assert_called_once_with(
            chat_id=4242, user_id=77, username="vasya", user_message=text,
        )

    async def test_short_text_does_not_schedule_extraction(self):
        node, _ = make_classify_node("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="Гениально)"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patched_side_effects() as (_, extract):
            await run_filter_stage(state, node)
        extract.assert_not_called()


class TestRouterToFilterStageWiring:
    """``broadcast_reply`` is produced by ``MessageRouter.__decide`` and consumed
    by the classify node purely through an untyped ``NotRequired`` state key — a
    typo on either side would fail open silently (the bot would answer as if
    this whole gate did not exist) and every other test would stay green,
    because each layer's tests mock the other layer's output instead of
    actually connecting them. This test flows the router's real return value
    into the filter stage's real input, with no hand-built intermediate state.
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
        node, _ = make_classify_node("NOT_ADDRESSED")
        with patched_side_effects():
            result = await run_filter_stage(state, node)

        assert result["should_respond"] is False
        assert result["drop_reason"] == "broadcast_not_addressed"
