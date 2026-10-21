"""Verdict enum tests: members behave as the plain labels the rest of the system stores."""

import json

import pytest

from src.pipeline.state import Verdict

LABELS = [
    "MEANINGFUL", "MEANINGLESS", "BANTER", "BOT_INSULT", "MEME_REQUEST",
    "GROUP_PROFILE_REQUEST", "NOT_ADDRESSED", "SHORTS", "SOCIAL_LINK",
]


@pytest.mark.parametrize("label", LABELS)
def test_verdict_member_equals_its_plain_string(label):
    assert Verdict[label] == label
    assert Verdict(label) is Verdict[label]


def test_verdict_has_exactly_the_known_labels():
    assert {member.value for member in Verdict} == set(LABELS)


def test_verdict_renders_as_plain_label_in_str_fstring_and_json():
    assert str(Verdict.BANTER) == "BANTER"
    assert f"{Verdict.BANTER}" == "BANTER"
    assert json.dumps(Verdict.BANTER) == '"BANTER"'
