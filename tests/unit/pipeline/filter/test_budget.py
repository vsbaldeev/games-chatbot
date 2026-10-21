"""charge_budget tests: attention-budget tiers, reply flags, emoji reactions."""

from unittest.mock import AsyncMock, patch

import pytest
from telegram import ReactionTypeEmoji

from src.config.prompts import GROUP_PROFILE_COOLDOWN_REPLIES
from src.pipeline import engagement_gate
from src.pipeline.filter.budget import (
    REACTION_POOL,
    charge_budget,
    build_reply_flags,
    send_reaction,
)
from src.pipeline.state import ContentKind, Verdict
from src.utils.ttl_gate import TtlGate
from tests.builders import make_incoming, make_state


class TestReactionPool:
    def test_reaction_pool_is_not_empty(self):
        assert len(REACTION_POOL) > 0

    def test_all_reactions_are_strings(self):
        assert all(isinstance(emoji, str) for emoji in REACTION_POOL)


class TestMemeRequestFlags:
    def test_full_tier_sets_meme_request(self):
        state = make_state(make_incoming())
        update = build_reply_flags(
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
        state = make_state(make_incoming())
        update = build_reply_flags(state, "MEME_REQUEST", tier)
        assert "meme_request" not in update
        assert update["wind_down"] is True

    def test_meme_request_weight_is_an_ordinary_message(self):
        assert engagement_gate.SIGNAL_WEIGHTS["MEME_REQUEST"] == 1.0


class TestGroupProfileRequestFlags:
    def test_full_tier_sets_group_profile_request(self):
        state = make_state(make_incoming(chat_id=2001))
        with patch("src.pipeline.filter.budget.group_profile_cooldown_gate", TtlGate(600)):
            update = build_reply_flags(
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
        state = make_state(make_incoming(chat_id=2002))
        update = build_reply_flags(state, "GROUP_PROFILE_REQUEST", tier)
        assert "group_profile_request" not in update
        assert update["wind_down"] is True

    def test_group_profile_request_weight_is_heavier_than_ordinary(self):
        assert engagement_gate.SIGNAL_WEIGHTS["GROUP_PROFILE_REQUEST"] > engagement_gate.SIGNAL_WEIGHTS["MEANINGFUL"]

    def test_second_request_within_cooldown_gets_canned_refusal(self):
        with patch("src.pipeline.filter.budget.group_profile_cooldown_gate", TtlGate(600)):
            first_state = make_state(make_incoming(chat_id=2003))
            first = build_reply_flags(
                first_state, "GROUP_PROFILE_REQUEST", engagement_gate.FULL_TIER
            )
            assert first["group_profile_request"] is True

            second_state = make_state(make_incoming(chat_id=2003))
            second = build_reply_flags(
                second_state, "GROUP_PROFILE_REQUEST", engagement_gate.FULL_TIER
            )
        assert second["should_respond"] is False
        assert second["response"] in GROUP_PROFILE_COOLDOWN_REPLIES
        assert second["content_kind"] == ContentKind.CAP_NOTICE
        assert "group_profile_request" not in second

    def test_different_chats_do_not_share_the_cooldown(self):
        with patch("src.pipeline.filter.budget.group_profile_cooldown_gate", TtlGate(600)):
            first_state = make_state(make_incoming(chat_id=2004))
            build_reply_flags(
                first_state, "GROUP_PROFILE_REQUEST", engagement_gate.FULL_TIER
            )
            other_state = make_state(make_incoming(chat_id=2005))
            other = build_reply_flags(
                other_state, "GROUP_PROFILE_REQUEST", engagement_gate.FULL_TIER
            )
        assert other["group_profile_request"] is True


class TestSendReaction:
    async def test_sends_reaction_from_pool(self):
        state = make_state(make_incoming())
        mock_bot = AsyncMock()
        state["context_types"].bot = mock_bot

        await send_reaction(state, REACTION_POOL)

        mock_bot.set_message_reaction.assert_called_once()
        sent_reaction = mock_bot.set_message_reaction.call_args.kwargs["reaction"][0]
        assert isinstance(sent_reaction, ReactionTypeEmoji)
        assert sent_reaction.emoji in REACTION_POOL

    async def test_sends_to_correct_chat_and_message(self):
        state = make_state(make_incoming(chat_id=9999, message_id=777))
        mock_bot = AsyncMock()
        state["context_types"].bot = mock_bot

        await send_reaction(state, REACTION_POOL)

        call_kwargs = mock_bot.set_message_reaction.call_args.kwargs
        assert call_kwargs["chat_id"] == 9999
        assert call_kwargs["message_id"] == 777

    async def test_bot_error_is_swallowed(self):
        state = make_state(make_incoming())
        mock_bot = AsyncMock()
        mock_bot.set_message_reaction.side_effect = Exception("Telegram API down")
        state["context_types"].bot = mock_bot

        await send_reaction(state, REACTION_POOL)


class TestChargeBudget:
    async def test_reads_verdict_from_state(self):
        state = make_state(
            make_incoming(), should_respond=True, filter_verdict="BANTER"
        )
        with patch(
            "src.pipeline.filter.budget.engagement_gate.register_signal",
            new=AsyncMock(return_value=engagement_gate.FULL_TIER),
        ):
            result = await charge_budget(state)
        assert result["filter_verdict"] == "BANTER"
        assert result["wind_down"] is True


def test_signal_weights_cover_exactly_the_chargeable_verdicts():
    assert set(engagement_gate.SIGNAL_WEIGHTS) == {
        Verdict.GROUP_PROFILE_REQUEST, Verdict.BOT_INSULT, Verdict.BANTER,
        Verdict.MEANINGLESS, Verdict.MEANINGFUL, Verdict.MEME_REQUEST,
    }
