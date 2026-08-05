"""
Part 2 — shared member-material gatherer.

Covers the pure formatting/selection helpers and the async ``gather`` composed
over mocked stores. The gatherer feeds both the comedian (autonomous humor) and
the offense auto-roast, so each section must degrade cleanly to empty.
"""

from unittest.mock import AsyncMock, patch

from src.agent.roast_material import (
    MAX_QUOTES,
    MAX_STAT_HIGHLIGHTS,
    MemberMaterial,
    format_member_material,
    gather_member_material,
    select_stat_highlights,
    truncate_quotes,
)

CHAT_ID = 1000
USER_ID = 42


class TestSelectStatHighlights:
    def test_skips_zero_and_unknown_stats(self):
        stats = {"duel_wins": 0, "unknown_stat": 99, "roasted_count": 3}
        highlights = select_stat_highlights(stats)
        assert len(highlights) == 1
        assert "3" in highlights[0]

    def test_sorted_descending_and_capped(self):
        stats = {
            "roasted_count": 1, "duel_wins": 2, "night_messages": 3,
            "sticker_messages": 4, "voice_messages": 5,
        }
        highlights = select_stat_highlights(stats)
        assert len(highlights) == MAX_STAT_HIGHLIGHTS
        assert "5" in highlights[0]  # highest-value stat first

    def test_empty_stats_yield_no_highlights(self):
        assert select_stat_highlights({}) == []


class TestTruncateQuotes:
    def test_caps_count(self):
        messages = [f"сообщение {index}" for index in range(10)]
        assert len(truncate_quotes(messages)) == MAX_QUOTES

    def test_truncates_long_quote(self):
        result = truncate_quotes(["а" * 500])
        assert result[0].endswith("…")
        assert len(result[0]) < 500

    def test_skips_blank_messages(self):
        assert truncate_quotes(["   ", ""]) == []


class TestFormatMemberMaterial:
    def test_empty_material_is_empty_string(self):
        material = MemberMaterial(username="vasya")
        assert material.is_empty is True
        assert format_member_material(material) == ""

    def test_includes_all_present_sections(self):
        material = MemberMaterial(
            username="vasya",
            facts=["любит доту"],
            quotes=["я лучший"],
            role={"tag": "Молчун", "reason": "молчит неделями"},
            stats=["прожарен раз: 3"],
        )
        block = format_member_material(material)
        assert "любит доту" in block
        assert "я лучший" in block
        assert "Молчун" in block
        assert "молчит неделями" in block
        assert "прожарен раз: 3" in block

    def test_role_without_reason_still_renders_tag(self):
        material = MemberMaterial(username="vasya", role={"tag": "Молчун", "reason": ""})
        block = format_member_material(material)
        assert "Молчун" in block
        assert "За что" not in block


class TestGatherMemberMaterial:
    async def test_composes_all_sources(self):
        with patch("src.agent.roast_material.get_facts", AsyncMock(return_value=["факт1"])), \
             patch("src.agent.roast_material.unified_messages.get_user_messages",
                   AsyncMock(return_value=["цитата"])), \
             patch("src.agent.roast_material.user_tags.get_tag",
                   AsyncMock(return_value={"tag": "Молчун", "reason": "тихий"})), \
             patch("src.agent.roast_material.achievements.get_user_stats",
                   AsyncMock(return_value={"roasted_count": 2})):
            material = await gather_member_material(CHAT_ID, USER_ID, "vasya")
        assert material.username == "vasya"
        assert material.facts == ["факт1"]
        assert material.quotes == ["цитата"]
        assert material.role == {"tag": "Молчун", "reason": "тихий"}
        assert material.stats == ["прожарен раз: 2"]
        assert material.is_empty is False

    async def test_missing_everything_yields_empty_material(self):
        with patch("src.agent.roast_material.get_facts", AsyncMock(return_value=[])), \
             patch("src.agent.roast_material.unified_messages.get_user_messages",
                   AsyncMock(return_value=[])), \
             patch("src.agent.roast_material.user_tags.get_tag", AsyncMock(return_value=None)), \
             patch("src.agent.roast_material.achievements.get_user_stats",
                   AsyncMock(return_value={})):
            material = await gather_member_material(CHAT_ID, USER_ID, "ghost")
        assert material.is_empty is True
