"""Gather the chat roster, split into members with material and unknowns.

A member with no facts, quotes, role or stats on record is excluded from
``materials_by_uid`` entirely — they are never sent to the LLM (see
generate.py), so an invented profile for them is impossible by construction.
Their line comes from a canned pool in render.py instead.
"""

import asyncio
from dataclasses import dataclass, field

from src import achievements
from src.agent.roast_material import MemberMaterial, gather_member_material


@dataclass
class ChatRoster:
    """One chat's members, split by whether there is any material to judge them on.

    Attributes:
        names_by_uid: Every known member's display name, for rendering.
        materials_by_uid: Non-empty dossiers, eligible for the LLM call.
        unknown_uids: Members with nothing on record.
    """

    names_by_uid: dict[int, str] = field(default_factory=dict)
    materials_by_uid: dict[int, MemberMaterial] = field(default_factory=dict)
    unknown_uids: list[int] = field(default_factory=list)


async def gather_roster(chat_id: int) -> ChatRoster:
    """Load every chat member's dossier and split by whether it is usable.

    Args:
        chat_id: Chat to profile.

    Returns:
        A ``ChatRoster`` with no members at all when the chat has none.
    """
    members = await achievements.get_chat_members(chat_id)
    if not members:
        return ChatRoster()
    names_by_uid = {user_id: name for user_id, name in members}
    materials = await asyncio.gather(*[
        gather_member_material(chat_id, user_id, names_by_uid[user_id])
        for user_id in names_by_uid
    ])
    materials_by_uid: dict[int, MemberMaterial] = {}
    unknown_uids: list[int] = []
    for user_id, material in zip(names_by_uid, materials):
        if material.is_empty:
            unknown_uids.append(user_id)
        else:
            materials_by_uid[user_id] = material
    return ChatRoster(
        names_by_uid=names_by_uid,
        materials_by_uid=materials_by_uid,
        unknown_uids=unknown_uids,
    )
