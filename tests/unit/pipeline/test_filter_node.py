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

from src.config.prompts import GROUP_PROFILE_COOLDOWN_REPLIES
from src.pipeline import engagement_gate
from src.pipeline.filter_node import (
    FILTER_SYSTEM,
    GROUP_PROFILE_MARKER_RE,
    REACTION_POOL,
    MeaninglessFilterNode,
    looks_like_request,
)
from src.utils.ttl_gate import TtlGate
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

    def test_mentions_meme_request_label(self):
        assert "MEME_REQUEST" in FILTER_SYSTEM

    def test_no_longer_routes_meme_requests_to_meaningful(self):
        """«скинь мем» used to be named as a MEANINGFUL example by the prompt."""
        assert "'скинь мем', 'кинь мемас'" in FILTER_SYSTEM

    def test_mentions_group_profile_request_label(self):
        assert "GROUP_PROFILE_REQUEST" in FILTER_SYSTEM

    def test_mentions_not_addressed_label(self):
        assert "NOT_ADDRESSED" in FILTER_SYSTEM

    def test_no_longer_asserts_the_message_is_addressed(self):
        """The premise made the addressee question unaskable (2026-08-18 incident)."""
        assert "is addressed to the bot" not in FILTER_SYSTEM

    def test_third_person_talk_is_named_as_not_addressed(self):
        assert "он говорит" in FILTER_SYSTEM

    def test_defaults_away_from_not_addressed_when_unsure(self):
        """Fail-open is preserved: silence is the riskier failure."""
        assert "never 'NOT_ADDRESSED'" in FILTER_SYSTEM


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
        assert result == {"should_respond": False, "drop_reason": "no_transcription"}

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
        assert result == {"should_respond": False, "drop_reason": "no_transcription"}


class TestRequestOverrideIgnoresMentions:
    """The question/request override judges what the user typed, not the @handle.

    An addressed message reaches the filter with its «@bot» prefix intact, so
    the handle used to take the leading-word slot and inflate the word count —
    silently disabling the override for every @mentioned short question.
    """

    @pytest.mark.parametrize("text", [
        "@zhora_bot о чем я говорю",
        "@zhora_bot что я описал",
        "@zhora_bot как это работает",
        "@zhora_bot переведи",
        "@zhora_bot скинь мем",
        "@zhora_bot ахаха что это было",
        "@zhora_bot @vasya кто прав",
    ])
    def test_mentioned_question_or_request_is_recognized(self, text):
        """Questions and imperatives behind an @handle still read as requests."""
        assert looks_like_request(text) is True

    @pytest.mark.parametrize("text", [
        "@zhora_bot ок",
        "@zhora_bot ахаха",
        "@zhora_bot бля",
        "@zhora_bot сам такой",
        "@zhora_bot ну ты и фрукт",
    ])
    def test_mentioned_short_reaction_is_still_not_a_request(self, text):
        """Genuine short reactions keep their MEANINGLESS/BANTER verdict."""
        assert looks_like_request(text) is False

    def test_handle_does_not_count_toward_substantive_word_count(self):
        """A six-word message plus a handle is not promoted by word count alone."""
        assert looks_like_request("@zhora_bot ну вот опять началось это всё") is False

    async def test_mentioned_question_survives_a_meaningless_verdict(self):
        """The reported bug: a mentioned question no longer drops to an emoji."""
        node, _ = make_node_with_mock_llm("MEANINGLESS")
        with patch("asyncio.create_task", side_effect=close_coroutine) as mock_create_task:
            state = make_state(
                make_incoming(raw_text="@zhora_bot о чем я говорю"), should_respond=True
            )
            result = await node(state)
        assert result["should_respond"] is True
        assert result["filter_verdict"] == "MEANINGFUL"
        mock_create_task.assert_not_called()


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

    async def test_meme_request_response_returns_meme_request(self):
        node, _ = make_node_with_mock_llm("MEME_REQUEST")
        result = await node._MeaninglessFilterNode__classify("скинь мем", FILTER_SYSTEM)
        assert result == "MEME_REQUEST"

    async def test_group_profile_request_response_returns_group_profile_request(self):
        node, _ = make_node_with_mock_llm("GROUP_PROFILE_REQUEST")
        result = await node._MeaninglessFilterNode__classify("оцени всем счастье", FILTER_SYSTEM)
        assert result == "GROUP_PROFILE_REQUEST"

    @pytest.mark.parametrize(
        "llm_response, expected",
        [
            ("MEANINGFUL", "MEANINGFUL"),
            ("MEANINGLESS", "MEANINGLESS"),
            ("PHOTO_REQUEST", "PHOTO_REQUEST"),
        ],
        ids=["meaningful", "meaningless", "photo-request"],
    )
    async def test_other_labels_are_not_swallowed_by_the_meme_check(self, llm_response, expected):
        """The MEME substring test must not capture the pre-existing labels."""
        node, _ = make_node_with_mock_llm(llm_response)
        result = await node._MeaninglessFilterNode__classify("текст", FILTER_SYSTEM)
        assert result == expected


class TestMemeRequestFlags:
    def test_full_tier_sets_meme_request(self):
        node = MeaninglessFilterNode()
        state = make_state(make_incoming())
        update = node._MeaninglessFilterNode__build_reply_flags(
            state, "MEME_REQUEST", engagement_gate.FULL_TIER
        )
        assert update["meme_request"] is True
        assert not update.get("wind_down")

    @pytest.mark.parametrize(
        "tier",
        [engagement_gate.BRUSH_OFF_TIER, engagement_gate.EMOJI_TIER],
        ids=["brush-off", "emoji"],
    )
    def test_lower_tiers_wind_down_instead_of_sending(self, tier):
        node = MeaninglessFilterNode()
        state = make_state(make_incoming())
        update = node._MeaninglessFilterNode__build_reply_flags(state, "MEME_REQUEST", tier)
        assert "meme_request" not in update
        assert update["wind_down"] is True

    def test_meme_request_weight_is_an_ordinary_message(self):
        assert engagement_gate.SIGNAL_WEIGHTS["MEME_REQUEST"] == 1.0


class TestGroupProfileMarkerRegex:
    @pytest.mark.parametrize(
        "text",
        [
            "оцени всем счастье от 1 до 10",
            "раздай всех роли из людей икс",
            "распредели каждому факультет",
            "rate everyone's happiness",
            "how does everybody rank",
            "assign a role to each of us",
        ],
    )
    def test_whole_chat_markers_match(self, text):
        assert GROUP_PROFILE_MARKER_RE.search(text)

    @pytest.mark.parametrize(
        "text",
        [
            "оцени Васю по шкале от 1 до 10",
            "кто из нас ты думаешь",
            "rate my happiness",
        ],
    )
    def test_single_person_requests_do_not_match(self, text):
        assert not GROUP_PROFILE_MARKER_RE.search(text)


class TestGroupProfileRequestDowngrade:
    async def test_no_whole_chat_marker_downgrades_to_meaningful(self):
        node, _ = make_node_with_mock_llm("GROUP_PROFILE_REQUEST")
        text = "оцени Васю по шкале от 1 до 10"
        state = make_state(make_incoming(raw_text=text))
        result = await node._MeaninglessFilterNode__classify_addressed(state, text)
        assert result == "MEANINGFUL"

    async def test_whole_chat_marker_present_keeps_the_label(self):
        node, _ = make_node_with_mock_llm("GROUP_PROFILE_REQUEST")
        text = "оцени всем счастье от 1 до 10"
        state = make_state(make_incoming(raw_text=text))
        result = await node._MeaninglessFilterNode__classify_addressed(state, text)
        assert result == "GROUP_PROFILE_REQUEST"


class TestGroupProfileRequestFlags:
    def test_full_tier_sets_group_profile_request(self):
        node = MeaninglessFilterNode()
        state = make_state(make_incoming(chat_id=2001))
        with patch("src.pipeline.filter_node.group_profile_cooldown_gate", TtlGate(600)):
            update = node._MeaninglessFilterNode__build_reply_flags(
                state, "GROUP_PROFILE_REQUEST", engagement_gate.FULL_TIER
            )
        assert update["group_profile_request"] is True
        assert not update.get("wind_down")

    @pytest.mark.parametrize(
        "tier",
        [engagement_gate.BRUSH_OFF_TIER, engagement_gate.EMOJI_TIER],
        ids=["brush-off", "emoji"],
    )
    def test_lower_tiers_wind_down_instead_of_sending(self, tier):
        node = MeaninglessFilterNode()
        state = make_state(make_incoming(chat_id=2002))
        update = node._MeaninglessFilterNode__build_reply_flags(state, "GROUP_PROFILE_REQUEST", tier)
        assert "group_profile_request" not in update
        assert update["wind_down"] is True

    def test_group_profile_request_weight_is_heavier_than_ordinary(self):
        assert engagement_gate.SIGNAL_WEIGHTS["GROUP_PROFILE_REQUEST"] > engagement_gate.SIGNAL_WEIGHTS["MEANINGFUL"]

    def test_second_request_within_cooldown_gets_canned_refusal(self):
        node = MeaninglessFilterNode()
        with patch("src.pipeline.filter_node.group_profile_cooldown_gate", TtlGate(600)):
            first_state = make_state(make_incoming(chat_id=2003))
            first = node._MeaninglessFilterNode__build_reply_flags(
                first_state, "GROUP_PROFILE_REQUEST", engagement_gate.FULL_TIER
            )
            assert first["group_profile_request"] is True

            second_state = make_state(make_incoming(chat_id=2003))
            second = node._MeaninglessFilterNode__build_reply_flags(
                second_state, "GROUP_PROFILE_REQUEST", engagement_gate.FULL_TIER
            )
        assert second["should_respond"] is False
        assert second["response"] in GROUP_PROFILE_COOLDOWN_REPLIES
        assert "group_profile_request" not in second

    def test_different_chats_do_not_share_the_cooldown(self):
        node = MeaninglessFilterNode()
        with patch("src.pipeline.filter_node.group_profile_cooldown_gate", TtlGate(600)):
            first_state = make_state(make_incoming(chat_id=2004))
            node._MeaninglessFilterNode__build_reply_flags(
                first_state, "GROUP_PROFILE_REQUEST", engagement_gate.FULL_TIER
            )
            other_state = make_state(make_incoming(chat_id=2005))
            other = node._MeaninglessFilterNode__build_reply_flags(
                other_state, "GROUP_PROFILE_REQUEST", engagement_gate.FULL_TIER
            )
        assert other["group_profile_request"] is True


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


def make_filter_node() -> MeaninglessFilterNode:
    with patch("src.pipeline.filter_node.make_filter_llm"):
        return MeaninglessFilterNode()


class TestYoutubeShortDispatchCharacterization:
    """Pins today's untested Shorts-trigger dispatch before this file's own
    social-link branch is added right beside it."""

    async def test_content_present_sets_shorts_verdict(self):
        node = make_filter_node()
        incoming = make_incoming(raw_text="https://www.youtube.com/shorts/abc")
        state = make_state(
            incoming, should_respond=True, response_trigger="youtube_short",
            youtube_short_content="[YouTube Shorts]\nsome content",
        )
        result = await node(state)
        assert result == {"filter_verdict": "SHORTS"}

    async def test_no_content_and_unaddressed_drops_silently(self):
        node = make_filter_node()
        telegram_message = MagicMock(text="https://www.youtube.com/shorts/abc", caption=None)
        telegram_message.reply_to_message = None
        incoming = make_incoming(
            raw_text="https://www.youtube.com/shorts/abc", telegram_message=telegram_message,
        )
        state = make_state(
            incoming, should_respond=True, response_trigger="youtube_short",
            youtube_short_content=None,
        )
        result = await node(state)
        assert result == {"should_respond": False, "drop_reason": "shorts_failed"}


class TestSocialLinkDispatch:
    async def test_content_present_sets_social_link_verdict(self):
        node = make_filter_node()
        incoming = make_incoming(raw_text="https://www.reddit.com/r/x/comments/abc/y/")
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_content="[Reddit r/x] «title»",
        )
        result = await node(state)
        assert result == {"filter_verdict": "SOCIAL_LINK"}

    async def test_no_content_and_unaddressed_drops_silently(self):
        node = make_filter_node()
        telegram_message = MagicMock(text="https://www.reddit.com/r/x/comments/abc/y/", caption=None)
        telegram_message.reply_to_message = None
        incoming = make_incoming(
            raw_text="https://www.reddit.com/r/x/comments/abc/y/", telegram_message=telegram_message,
        )
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_content=None,
        )
        result = await node(state)
        assert result == {"should_respond": False, "drop_reason": "social_link_failed"}

    async def test_no_content_and_addressed_gets_canned_reply(self):
        node = make_filter_node()
        telegram_message = MagicMock(
            text="@testbot https://www.reddit.com/r/x/comments/abc/y/", caption=None,
        )
        telegram_message.reply_to_message = None
        incoming = make_incoming(
            raw_text="@testbot https://www.reddit.com/r/x/comments/abc/y/",
            telegram_message=telegram_message,
        )
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_content=None,
        )
        result = await node(state)
        assert result["should_respond"] is False
        assert result["response"]  # one of SOCIAL_LINK_FAILED_REPLIES
