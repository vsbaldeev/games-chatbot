"""Group-profile rendering tests.

Pins the two block shapes: a decided verdict (LLM-sourced) versus an unknown
member (canned pool, in code — never sent to the LLM in the first place).
"""

from src.config.prompts import GROUP_PROFILE_UNKNOWN_LINES
from src.group_profile import render


class TestRenderVerdictBlock:
    def test_includes_name_verdict_and_reason(self):
        block = render.render_verdict_block("alice", {"verdict": "Циклоп", "reason": "видит только вперёд"})
        assert block == "@alice — Циклоп\nвидит только вперёд"

    def test_omits_reason_line_when_absent(self):
        block = render.render_verdict_block("alice", {"verdict": "7/10", "reason": ""})
        assert block == "@alice — 7/10"


class TestRenderUnknownBlock:
    def test_uses_one_of_the_canned_lines(self):
        block = render.render_unknown_block("bob")
        assert block.startswith("@bob — ")
        line = block.removeprefix("@bob — ")
        assert line in GROUP_PROFILE_UNKNOWN_LINES


class TestBuildMessage:
    def test_combines_verdicts_and_unknowns(self):
        verdicts = {1: {"verdict": "Циклоп", "reason": "r1"}}
        text = render.build_message(verdicts, [2], {1: "alice", 2: "bob"})
        assert "@alice — Циклоп" in text
        assert "@bob — " in text

    def test_falls_back_to_user_id_when_name_missing(self):
        verdicts = {99: {"verdict": "X", "reason": ""}}
        text = render.build_message(verdicts, [], {})
        assert "@99 — X" in text

    def test_nothing_to_show_returns_empty_string(self):
        assert render.build_message({}, [], {}) == ""
