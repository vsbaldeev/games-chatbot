"""Filter stage tests: the four filter nodes wired by ``add_filter_stage``.

Runs the stage as a small compiled graph (``START → … → END``), so these tests
cover the real routing between link_check, media_check, classify and budget.

Only the wiring between nodes is asserted here: which path a message takes and
which side effects it triggers. Each node's own behaviour is tested beside it
(``test_classify.py``, ``test_budget.py``, ``test_media_check.py``,
``test_link_check.py``).

Patching ``asyncio.create_task`` globally breaks LangGraph's own scheduling, so
emoji reactions are observed on a mocked bot and passive fact extraction is
patched at the memory writer's boundary.
"""

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from langgraph.graph import END, START, StateGraph

from src import config
from src.pipeline import engagement_gate
from src.pipeline.filter.classify import ClassifyNode
from src.pipeline.graph import add_filter_stage
from src.pipeline.router import MessageRouter
from src.pipeline.state import BotState, ContentKind
from tests.builders import make_classify_node, make_incoming, make_state, make_telegram_message


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
def patched_extraction():
    """Patch passive fact extraction (the memory writer's boundary); yields the mock."""
    with patch("src.pipeline.filter.classify.extract_and_save", new=AsyncMock()) as extract:
        yield extract


def attach_bot(state: dict) -> AsyncMock:
    """Give the state an async bot so emoji reactions are observable; returns it."""
    bot = AsyncMock()
    state["context_types"].bot = bot
    return bot


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
    async def test_meaningless_text_ends_the_run_with_a_reaction(self):
        node, _ = make_classify_node("MEANINGLESS")
        state = make_state(make_incoming(raw_text="ахаха"), should_respond=True)
        bot = attach_bot(state)
        result = await run_filter_stage(state, node)
        assert result == {
            "should_respond": False,
            "drop_reason": "meaningless",
            "filter_verdict": "MEANINGLESS",
            "engagement_tier": 1,
        }
        bot.set_message_reaction.assert_called_once()

    async def test_meaningful_text_continues_without_a_reaction(self):
        node, _ = make_classify_node("MEANINGFUL")
        state = make_state(make_incoming(raw_text="расскажи про GTA 6"), should_respond=True)
        bot = attach_bot(state)
        result = await run_filter_stage(state, node)
        assert result == {
            "should_respond": True,
            "filter_verdict": "MEANINGFUL",
            "engagement_tier": 1,
        }
        bot.set_message_reaction.assert_not_called()

    async def test_mentioned_question_survives_a_meaningless_verdict(self):
        """A mentioned question no longer drops to an emoji."""
        node, _ = make_classify_node("MEANINGLESS")
        state = make_state(
            make_incoming(raw_text="@zhora_bot о чем я говорю"), should_respond=True
        )
        bot = attach_bot(state)
        result = await run_filter_stage(state, node)
        assert result["should_respond"] is True
        assert result["filter_verdict"] == "MEANINGFUL"
        bot.set_message_reaction.assert_not_called()


class TestMediaMessages:
    async def test_no_transcription_gets_canned_reply_and_no_reaction(self):
        node, _ = make_classify_node()
        state = make_state(
            make_incoming(media_type="voice", processed_text=None),
            should_respond=True,
            response_trigger="explicit",
        )
        bot = attach_bot(state)
        result = await run_filter_stage(state, node)
        assert result["should_respond"] is False
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE
        bot.set_message_reaction.assert_not_called()

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
        bot = attach_bot(state)
        with patched_extraction() as extract:
            result = await run_filter_stage(state, node)
        assert result["should_respond"] is False
        assert result["drop_reason"] == "broadcast_not_addressed"
        # Only the passive extraction fires (the text is long enough) — no
        # emoji reaction, no reply.
        extract.assert_called_once()
        bot.set_message_reaction.assert_not_called()

    async def test_not_addressed_charges_no_attention_budget(self):
        """The bot was never addressed — charging the user would wind them
        down for talking to their friends."""
        node, _ = make_classify_node("NOT_ADDRESSED")
        state = make_state(
            make_incoming(raw_text="Гениально)"),
            should_respond=True, response_trigger="explicit", broadcast_reply=True,
        )
        with patched_extraction(), patch_register_signal() as mock_register:
            await run_filter_stage(state, node)
        mock_register.assert_not_called()


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
        with patched_extraction() as extract:
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
        with patched_extraction() as extract:
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
        with patched_extraction():
            result = await run_filter_stage(state, node)

        assert result["should_respond"] is False
        assert result["drop_reason"] == "broadcast_not_addressed"
