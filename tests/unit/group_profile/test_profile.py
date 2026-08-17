"""Group-profile orchestration tests: roster -> verdicts -> rendered message."""

from unittest.mock import AsyncMock, patch

from src.group_profile import profile
from src.group_profile.roster import ChatRoster

GATHER_ROSTER = "src.group_profile.profile.gather_roster"
GENERATE_VERDICTS = "src.group_profile.profile.generate_verdicts"
FILL_MISSING = "src.group_profile.profile.fill_missing_verdicts"

CHAT_ID = 1000


class TestRunGroupProfile:
    async def test_no_members_at_all_returns_none(self):
        empty_roster = ChatRoster()
        with patch(GATHER_ROSTER, new_callable=AsyncMock, return_value=empty_roster):
            result = await profile.run_group_profile(CHAT_ID, "тема")
        assert result is None

    async def test_renders_message_from_verdicts_and_unknowns(self):
        roster = ChatRoster(
            names_by_uid={1: "alice", 2: "bob"},
            materials_by_uid={1: object()},
            unknown_uids=[2],
        )
        verdicts = {1: {"verdict": "Циклоп", "reason": "r1"}}
        with patch(GATHER_ROSTER, new_callable=AsyncMock, return_value=roster), \
             patch(GENERATE_VERDICTS, new_callable=AsyncMock, return_value=verdicts), \
             patch(FILL_MISSING, new_callable=AsyncMock, return_value=verdicts):
            result = await profile.run_group_profile(CHAT_ID, "роли из Людей Икс")
        assert "@alice — Циклоп" in result
        assert "@bob — " in result

    async def test_all_unknown_still_produces_a_message(self):
        roster = ChatRoster(names_by_uid={1: "alice"}, materials_by_uid={}, unknown_uids=[1])
        with patch(GATHER_ROSTER, new_callable=AsyncMock, return_value=roster), \
             patch(GENERATE_VERDICTS, new_callable=AsyncMock, return_value={}), \
             patch(FILL_MISSING, new_callable=AsyncMock, return_value={}):
            result = await profile.run_group_profile(CHAT_ID, "тема")
        assert "@alice — " in result
