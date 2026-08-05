"""
Part 5 — HumorNode (autonomous-humor integration node).

Reached only when the gate fires. It gathers the live conversation + participant
material, asks the comedian, and either sets state["response"] (so run_pipeline
delivers it) or stays silent. Errors fail safe to silence. These tests pin the
act / abstain / error paths, the gate side-effects, and the context helpers.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.comedian import ComedianDecision
from src.agent.roast_material import MemberMaterial
from src.pipeline import humor_gate
from src.pipeline.humor_node import (
    HumorNode,
    distinct_participants,
    gather_participants_material,
    render_conversation,
)

CHAT_ID = 777
BOT_ID = 999


@pytest.fixture(autouse=True)
def clear_gate_state():
    humor_gate.messages_since_joke.clear()
    humor_gate.last_joke_time.clear()
    yield
    humor_gate.messages_since_joke.clear()
    humor_gate.last_joke_time.clear()


class FakeAgent:
    def __init__(self, *, decision=None, error=None):
        self.decision = decision
        self.error = error

    async def decide(self, conversation, material):
        if self.error:
            raise self.error
        return self.decision


def state() -> dict:
    return {"incoming": {"chat_id": CHAT_ID, "message_id": 5}}


class TestRenderConversation:
    def test_renders_oldest_first_with_id_markers(self):
        recent = [  # get_recent is newest-first
            {"message_id": 12, "username": "b", "media_type": "text", "content": "второе"},
            {"message_id": 11, "username": "a", "media_type": "text", "content": "первое"},
        ]
        rendered = render_conversation(recent)
        assert rendered.index("первое") < rendered.index("второе")
        assert "[#11]" in rendered
        assert "[#12]" in rendered


class TestDistinctParticipants:
    def test_dedupes_excludes_bot_and_caps(self):
        recent = [
            {"user_id": 1, "username": "a"},
            {"user_id": BOT_ID, "username": "bot"},
            {"user_id": 1, "username": "a"},
            {"user_id": 2, "username": "b"},
            {"user_id": 3, "username": "c"},
            {"user_id": 4, "username": "d"},
            {"user_id": 5, "username": "e"},
        ]
        participants = distinct_participants(recent, BOT_ID)
        ids = [user_id for user_id, _ in participants]
        assert BOT_ID not in ids
        assert ids == [1, 2, 3, 4]  # deduped, capped at MAX_PARTICIPANTS


class TestGatherParticipantsMaterial:
    async def test_skips_empty_and_formats_blocks(self):
        recent = [{"user_id": 1, "username": "vasya"}, {"user_id": 2, "username": "ghost"}]

        async def fake_gather(chat_id, user_id, username):
            if username == "vasya":
                return MemberMaterial(username="vasya", facts=["любит доту"])
            return MemberMaterial(username="ghost")

        with patch("src.pipeline.humor_node.gather_member_material", side_effect=fake_gather):
            block = await gather_participants_material(CHAT_ID, recent, BOT_ID)
        assert "@vasya" in block
        assert "любит доту" in block
        assert "ghost" not in block


class TestHumorNodeCall:
    async def test_acts_sets_response_and_stamps_cooldown(self):
        agent = FakeAgent(decision=ComedianDecision(act=True, register="light", text="Кто проспорил?"))
        node = HumorNode(agent)
        with patch("src.pipeline.humor_node.unified_messages.get_recent", AsyncMock(return_value=[])), \
             patch("src.pipeline.humor_node.gather_participants_material", AsyncMock(return_value="")):
            result = await node(state())
        assert result == {
            "response": "Кто проспорил?",
            "response_trigger": "humor",
            "humor_reply_to_msg_id": None,
        }
        assert humor_gate.messages_since_joke[CHAT_ID] == 0
        assert CHAT_ID in humor_gate.last_joke_time

    async def test_abstain_stays_silent_and_marks_considered(self):
        agent = FakeAgent(decision=ComedianDecision.abstain())
        node = HumorNode(agent)
        with patch("src.pipeline.humor_node.unified_messages.get_recent", AsyncMock(return_value=[])), \
             patch("src.pipeline.humor_node.gather_participants_material", AsyncMock(return_value="")):
            result = await node(state())
        assert result == {}
        assert humor_gate.messages_since_joke[CHAT_ID] == 0
        assert CHAT_ID not in humor_gate.last_joke_time  # no cooldown without a sent joke

    async def test_error_fails_safe_to_silence(self):
        agent = FakeAgent(error=RuntimeError("groq down"))
        node = HumorNode(agent)
        with patch("src.pipeline.humor_node.unified_messages.get_recent", AsyncMock(return_value=[])), \
             patch("src.pipeline.humor_node.gather_participants_material", AsyncMock(return_value="")):
            result = await node(state())
        assert result == {}
        assert CHAT_ID not in humor_gate.last_joke_time
