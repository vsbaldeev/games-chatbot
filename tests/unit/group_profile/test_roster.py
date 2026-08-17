"""Chat-roster gathering tests.

Covers the split between members with real material (eligible for the LLM
call) and unknown members (excluded from it entirely, rendered from a canned
pool instead) — the guarantee that makes an invented profile for a ghost
member impossible by construction.
"""

from unittest.mock import AsyncMock, patch

from src.agent.roast_material import MemberMaterial
from src.group_profile import roster

GET_MEMBERS = "src.group_profile.roster.achievements.get_chat_members"
GATHER_MATERIAL = "src.group_profile.roster.gather_member_material"

CHAT_ID = 1000


def material_with_facts(username: str) -> MemberMaterial:
    return MemberMaterial(username=username, facts=["играет по ночам"])


def empty_material(username: str) -> MemberMaterial:
    return MemberMaterial(username=username)


class TestGatherRoster:
    async def test_splits_members_by_whether_material_is_empty(self):
        members = [(1, "alice"), (2, "bob")]
        materials = [material_with_facts("alice"), empty_material("bob")]
        with patch(GET_MEMBERS, new_callable=AsyncMock, return_value=members), \
             patch(GATHER_MATERIAL, new_callable=AsyncMock, side_effect=materials):
            result = await roster.gather_roster(CHAT_ID)

        assert set(result.materials_by_uid) == {1}
        assert result.unknown_uids == [2]
        assert result.names_by_uid == {1: "alice", 2: "bob"}

    async def test_no_members_returns_empty_roster(self):
        with patch(GET_MEMBERS, new_callable=AsyncMock, return_value=[]), \
             patch(GATHER_MATERIAL, new_callable=AsyncMock) as mock_gather:
            result = await roster.gather_roster(CHAT_ID)

        assert result.materials_by_uid == {}
        assert result.unknown_uids == []
        mock_gather.assert_not_awaited()

    async def test_all_members_have_material(self):
        members = [(1, "alice"), (2, "bob")]
        materials = [material_with_facts("alice"), material_with_facts("bob")]
        with patch(GET_MEMBERS, new_callable=AsyncMock, return_value=members), \
             patch(GATHER_MATERIAL, new_callable=AsyncMock, side_effect=materials):
            result = await roster.gather_roster(CHAT_ID)

        assert set(result.materials_by_uid) == {1, 2}
        assert result.unknown_uids == []
