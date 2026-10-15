"""Deriving a message's content kind and its question from the final pipeline state."""

import pytest

from src.message_content.kinds import content_kind_for, question_text
from src.pipeline.state import ContentKind
from tests.builders import make_incoming, make_state


def state_with(**overrides) -> dict:
    return make_state(make_incoming(), should_respond=True, response_trigger="explicit", **overrides)


class TestContentKindFor:
    def test_plain_explicit_reply_is_a_response(self):
        assert content_kind_for(state_with()) == ContentKind.RESPONSE

    def test_canned_kind_set_by_a_node_wins(self):
        state = state_with(content_kind=ContentKind.CAP_NOTICE, wind_down=True)
        assert content_kind_for(state) == ContentKind.CAP_NOTICE

    def test_wind_down_reply(self):
        assert content_kind_for(state_with(wind_down=True)) == ContentKind.WIND_DOWN_REPLY

    def test_insult_reply(self):
        assert content_kind_for(state_with(is_bot_insult=True)) == ContentKind.INSULT_REPLY

    def test_wound_down_insult_is_a_wind_down_reply(self):
        """A wound-down insult carries both flags; the wind-down directive is what the LLM obeyed."""
        state = state_with(is_bot_insult=True, wind_down=True)
        assert content_kind_for(state) == ContentKind.WIND_DOWN_REPLY

    @pytest.mark.parametrize("trigger", ["youtube_short", "social_link"])
    def test_link_triggers_are_link_summaries(self, trigger):
        state = make_state(make_incoming(), should_respond=True, response_trigger=trigger)
        assert content_kind_for(state) == ContentKind.LINK_SUMMARY

    def test_wind_down_beats_a_link_trigger(self):
        state = make_state(make_incoming(), response_trigger="social_link", wind_down=True)
        assert content_kind_for(state) == ContentKind.WIND_DOWN_REPLY

    def test_missing_trigger_defaults_to_response(self):
        state = {"incoming": make_incoming()}
        assert content_kind_for(state) == ContentKind.RESPONSE


class TestQuestionText:
    def test_prefers_the_transcript(self):
        incoming = make_incoming(raw_text="@bot", processed_text="  когда выйдет GTA 6?  ", media_type="voice")
        assert question_text(make_state(incoming)) == "когда выйдет GTA 6?"

    def test_falls_back_to_raw_text(self):
        incoming = make_incoming(raw_text="когда выйдет GTA 6?", processed_text=None)
        assert question_text(make_state(incoming)) == "когда выйдет GTA 6?"

    def test_blank_when_both_are_missing(self):
        incoming = make_incoming(raw_text=None, processed_text=None)
        assert question_text(make_state(incoming)) == ""
