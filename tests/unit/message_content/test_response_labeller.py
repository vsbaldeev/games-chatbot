"""Response labelling: payload assembly and strict response parsing.

Parsing is deliberately unforgiving, like the reply labeller's: an answer
outside the contract returns None, which the recorder stores as UNCLEAR
rather than a guess.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.feedback.reply_labeller import LabellingCallFailed
from src.message_content.response_labeller import (
    RESPONSE_LABELS,
    build_payload,
    label_response,
    parse_label,
)

KEY_TARGET = "src.message_content.response_labeller.config.OPENROUTER_API_KEY"
LLM_TARGET = "src.message_content.response_labeller.make_labeller_llm"
INVOKE_TARGET = "src.message_content.response_labeller.ainvoke_with_backoff"


class TestBuildPayload:
    def test_is_a_json_object_with_question_and_response(self):
        payload = build_payload(question="a", response="b")
        assert json.loads(payload) == {"question": "a", "response": "b"}

    def test_cyrillic_reaches_the_model_unescaped(self):
        payload = build_payload(question="когда выйдет GTA 6?", response="19 ноября")
        assert "когда выйдет GTA 6?" in payload
        assert "\\u" not in payload


class TestParseLabel:
    @pytest.mark.parametrize("label", sorted(RESPONSE_LABELS))
    def test_every_vocabulary_label_is_accepted(self, label):
        assert parse_label(json.dumps({"label": label})) == label

    def test_vocabulary_is_the_four_spec_labels(self):
        assert RESPONSE_LABELS == {"ANSWERED", "REFUSED", "NON_ANSWER", "UNCLEAR"}

    def test_unknown_label_is_rejected(self):
        assert parse_label(json.dumps({"label": "MAYBE"})) is None

    def test_key_typo_is_rejected(self):
        assert parse_label(json.dumps({"lable": "ANSWERED"})) is None

    def test_extra_key_is_rejected(self):
        assert parse_label(json.dumps({"label": "ANSWERED", "why": "ok"})) is None

    def test_array_instead_of_object_is_rejected(self):
        assert parse_label(json.dumps([{"label": "ANSWERED"}])) is None

    def test_list_valued_label_is_rejected_not_raised(self):
        assert parse_label(json.dumps({"label": ["ANSWERED"]})) is None

    def test_non_json_is_rejected(self):
        assert parse_label("it answered the question") is None

    def test_empty_response_is_rejected(self):
        assert parse_label("") is None

    def test_markdown_fenced_json_is_accepted(self):
        assert parse_label('```json\n{"label": "REFUSED"}\n```') == "REFUSED"

    def test_non_string_content_is_rejected_not_raised(self):
        assert parse_label([{"type": "text", "text": '{"label": "ANSWERED"}'}]) is None


class TestLabelResponse:
    async def test_missing_api_key_raises_call_failed(self):
        with patch(KEY_TARGET, ""):
            with pytest.raises(LabellingCallFailed):
                await label_response(question="a", response="b")

    async def test_transport_error_raises_call_failed(self):
        with patch(KEY_TARGET, "key"), patch(LLM_TARGET, MagicMock()), \
             patch(INVOKE_TARGET, AsyncMock(side_effect=RuntimeError("boom"))):
            with pytest.raises(LabellingCallFailed):
                await label_response(question="a", response="b")

    async def test_returns_parsed_label(self):
        answer = MagicMock(content='{"label": "NON_ANSWER"}')
        with patch(KEY_TARGET, "key"), patch(LLM_TARGET, MagicMock()), \
             patch(INVOKE_TARGET, AsyncMock(return_value=answer)):
            assert await label_response(question="a", response="b") == "NON_ANSWER"

    async def test_out_of_contract_answer_returns_none(self):
        answer = MagicMock(content="no idea")
        with patch(KEY_TARGET, "key"), patch(LLM_TARGET, MagicMock()), \
             patch(INVOKE_TARGET, AsyncMock(return_value=answer)):
            assert await label_response(question="a", response="b") is None

    async def test_uses_the_response_label_token_budget(self):
        answer = MagicMock(content='{"label": "ANSWERED"}')
        make_llm = MagicMock()
        with patch(KEY_TARGET, "key"), patch(LLM_TARGET, make_llm), \
             patch(INVOKE_TARGET, AsyncMock(return_value=answer)):
            await label_response(question="a", response="b")
        make_llm.assert_called_once()
        assert make_llm.call_args.args[0] == 64
