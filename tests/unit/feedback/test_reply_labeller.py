"""Single-reply labelling: payload assembly and strict response parsing.

Parsing is deliberately unforgiving. src/config/models.py records a previous
incident where a model emitted a "veredict" key typo that silently dropped a
member from a result; here an unexpected shape returns None, which the caller
counts as unlabelled rather than storing a guess.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.feedback.reply_labeller import (
    LabellingCallFailed,
    build_payload,
    label_reply,
    parse_label,
)


class TestBuildPayload:
    def test_includes_both_messages_and_the_source(self):
        payload = build_payload(
            bot_message="Elden Ring вышел в 2021", reply="ну и бред ты несешь", source="pipeline",
        )
        assert "ну и бред ты несешь" in payload
        assert "Elden Ring вышел в 2021" in payload
        assert "pipeline" in payload

    def test_is_a_json_object(self):
        payload = build_payload(bot_message="a", reply="b", source="meme")
        assert json.loads(payload) == {"bot_message": "a", "source": "meme", "reply": "b"}


class TestParseLabel:
    @pytest.mark.parametrize("label", ["POSITIVE", "NEGATIVE", "NEUTRAL", "UNCLEAR"])
    def test_every_vocabulary_label_is_accepted(self, label):
        assert parse_label(json.dumps({"reaction": label})) == label

    def test_media_is_out_of_contract(self):
        assert parse_label(json.dumps({"reaction": "MEDIA"})) is None

    def test_unknown_label_is_rejected(self):
        assert parse_label(json.dumps({"reaction": "ANGRY"})) is None

    def test_key_typo_is_rejected(self):
        assert parse_label(json.dumps({"veredict": "POSITIVE"})) is None

    def test_extra_key_is_rejected(self):
        assert parse_label(json.dumps({"reaction": "NEGATIVE", "cause": "TONE"})) is None

    def test_array_instead_of_object_is_rejected(self):
        assert parse_label(json.dumps([{"reaction": "POSITIVE"}])) is None

    def test_non_json_is_rejected(self):
        assert parse_label("I think it is positive") is None

    def test_empty_response_is_rejected(self):
        assert parse_label("") is None

    def test_markdown_fenced_json_is_parsed(self):
        assert parse_label('```json\n{"reaction": "NEUTRAL"}\n```') == "NEUTRAL"


class TestLabelReply:
    async def test_missing_api_key_raises_call_failed(self):
        with patch("src.feedback.reply_labeller.config.OPENROUTER_API_KEY", ""):
            with pytest.raises(LabellingCallFailed):
                await label_reply(bot_message="a", reply="b", source="pipeline")

    async def test_transport_error_raises_call_failed(self):
        with patch("src.feedback.reply_labeller.config.OPENROUTER_API_KEY", "key"), \
             patch("src.feedback.reply_labeller.make_labeller_llm", MagicMock()), \
             patch("src.feedback.reply_labeller.ainvoke_with_backoff",
                   AsyncMock(side_effect=RuntimeError("boom"))):
            with pytest.raises(LabellingCallFailed):
                await label_reply(bot_message="a", reply="b", source="pipeline")

    async def test_returns_parsed_label(self):
        response = MagicMock(content='{"reaction": "POSITIVE"}')
        with patch("src.feedback.reply_labeller.config.OPENROUTER_API_KEY", "key"), \
             patch("src.feedback.reply_labeller.make_labeller_llm", MagicMock()), \
             patch("src.feedback.reply_labeller.ainvoke_with_backoff", AsyncMock(return_value=response)):
            assert await label_reply(bot_message="a", reply="b", source="pipeline") == "POSITIVE"

    async def test_unparseable_answer_returns_none(self):
        response = MagicMock(content="no idea")
        with patch("src.feedback.reply_labeller.config.OPENROUTER_API_KEY", "key"), \
             patch("src.feedback.reply_labeller.make_labeller_llm", MagicMock()), \
             patch("src.feedback.reply_labeller.ainvoke_with_backoff", AsyncMock(return_value=response)):
            assert await label_reply(bot_message="a", reply="b", source="pipeline") is None


class TestParseLabelHostileShapes:
    def test_list_valued_reaction_is_rejected_not_raised(self):
        assert parse_label(json.dumps({"reaction": ["POSITIVE"]})) is None

    def test_dict_valued_reaction_is_rejected_not_raised(self):
        assert parse_label(json.dumps({"reaction": {"label": "POSITIVE"}})) is None

    async def test_content_block_list_answer_is_unparseable_not_raised(self):
        response = MagicMock(content=[{"type": "text", "text": '{"reaction": "POSITIVE"}'}])
        with patch("src.feedback.reply_labeller.config.OPENROUTER_API_KEY", "key"), \
             patch("src.feedback.reply_labeller.make_labeller_llm", MagicMock()), \
             patch("src.feedback.reply_labeller.ainvoke_with_backoff", AsyncMock(return_value=response)):
            assert await label_reply(bot_message="a", reply="b", source="pipeline") is None
