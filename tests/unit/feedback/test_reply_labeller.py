"""Batched reply labelling: prompt assembly and strict response parsing.

Parsing is deliberately unforgiving. src/config/models.py records a previous
incident where a model emitted a "veredict" key typo that silently dropped a
member from a result; here an unexpected shape leaves the row unlabelled
instead, so the job retries it rather than storing a guess.
"""

import json

from src.feedback.reply_labeller import (
    build_batch_payload,
    parse_label_batch,
)


def make_row(
    *,
    reply_message_id: int = 7,
    reply_content: str = "ну и бред ты несешь",
    bot_content: str = "Elden Ring вышел в 2021",
    source: str = "pipeline",
    trigger: str = "explicit",
) -> dict:
    return {
        "chat_id": 1000,
        "reply_message_id": reply_message_id,
        "media_type": "text",
        "reply_content": reply_content,
        "bot_content": bot_content,
        "source": source,
        "trigger": trigger,
    }


class TestBuildBatchPayload:
    def test_includes_reply_id_so_the_model_can_key_its_answer(self):
        payload = build_batch_payload([make_row(reply_message_id=42)])
        assert "42" in payload

    def test_includes_both_messages_and_the_source(self):
        payload = build_batch_payload([make_row()])
        assert "ну и бред ты несешь" in payload
        assert "Elden Ring вышел в 2021" in payload
        assert "pipeline" in payload

    def test_every_row_appears(self):
        payload = build_batch_payload([
            make_row(reply_message_id=1, reply_content="первый"),
            make_row(reply_message_id=2, reply_content="второй"),
        ])
        assert "первый" in payload
        assert "второй" in payload


class TestParseLabelBatch:
    def test_valid_single_item(self):
        raw = json.dumps([{"id": 7, "reaction": "NEGATIVE", "cause": "FACTUAL"}])
        assert parse_label_batch(raw, {7}) == {7: ("NEGATIVE", "FACTUAL")}

    def test_valid_multiple_items(self):
        raw = json.dumps([
            {"id": 1, "reaction": "POSITIVE", "cause": None},
            {"id": 2, "reaction": "NEGATIVE", "cause": "INVENTED_LIFE"},
        ])
        assert parse_label_batch(raw, {1, 2}) == {
            1: ("POSITIVE", None),
            2: ("NEGATIVE", "INVENTED_LIFE"),
        }

    def test_cause_on_a_positive_reaction_is_discarded(self):
        raw = json.dumps([{"id": 7, "reaction": "POSITIVE", "cause": "TONE"}])
        assert parse_label_batch(raw, {7}) == {7: ("POSITIVE", None)}

    def test_missing_cause_on_a_negative_becomes_unclear(self):
        raw = json.dumps([{"id": 7, "reaction": "NEGATIVE", "cause": None}])
        assert parse_label_batch(raw, {7}) == {7: ("NEGATIVE", "UNCLEAR")}

    def test_model_returned_media_is_dropped(self):
        """MEDIA is a rule label. A model emitting it is out of contract."""
        raw = json.dumps([{"id": 7, "reaction": "MEDIA", "cause": None}])
        assert parse_label_batch(raw, {7}) == {}

    def test_unknown_id_is_dropped(self):
        raw = json.dumps([{"id": 999, "reaction": "POSITIVE", "cause": None}])
        assert parse_label_batch(raw, {7}) == {}

    def test_duplicate_id_drops_the_whole_batch(self):
        raw = json.dumps([
            {"id": 7, "reaction": "POSITIVE", "cause": None},
            {"id": 7, "reaction": "NEGATIVE", "cause": "TONE"},
        ])
        assert parse_label_batch(raw, {7}) == {}

    def test_key_typo_drops_that_item_only(self):
        raw = json.dumps([
            {"id": 1, "veredict": "POSITIVE", "cause": None},
            {"id": 2, "reaction": "NEUTRAL", "cause": None},
        ])
        assert parse_label_batch(raw, {1, 2}) == {2: ("NEUTRAL", None)}

    def test_unknown_reaction_is_dropped(self):
        raw = json.dumps([{"id": 7, "reaction": "ANGRY", "cause": None}])
        assert parse_label_batch(raw, {7}) == {}

    def test_unknown_cause_becomes_unclear(self):
        raw = json.dumps([{"id": 7, "reaction": "NEGATIVE", "cause": "BOREDOM"}])
        assert parse_label_batch(raw, {7}) == {7: ("NEGATIVE", "UNCLEAR")}

    def test_non_json_yields_nothing(self):
        assert parse_label_batch("I think the first one is a rejection", {7}) == {}

    def test_json_object_instead_of_array_yields_nothing(self):
        assert parse_label_batch(json.dumps({"id": 7, "reaction": "POSITIVE"}), {7}) == {}

    def test_empty_response_yields_nothing(self):
        assert parse_label_batch("", {7}) == {}

    def test_markdown_fenced_json_is_parsed(self):
        raw = '```json\n[{"id": 7, "reaction": "NEUTRAL", "cause": null}]\n```'
        assert parse_label_batch(raw, {7}) == {7: ("NEUTRAL", None)}

    def test_neutral_never_carries_a_cause(self):
        raw = json.dumps([{"id": 7, "reaction": "NEUTRAL", "cause": "TONE"}])
        assert parse_label_batch(raw, {7}) == {7: ("NEUTRAL", None)}

    def test_unclear_never_carries_a_cause(self):
        raw = json.dumps([{"id": 7, "reaction": "UNCLEAR", "cause": "PERSONA"}])
        assert parse_label_batch(raw, {7}) == {7: ("UNCLEAR", None)}

    def test_non_integer_id_is_dropped(self):
        raw = json.dumps([{"id": "seven", "reaction": "NEUTRAL", "cause": None}])
        assert parse_label_batch(raw, {7}) == {}
