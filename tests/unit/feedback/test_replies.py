"""Reply-to-bot thread tracking: credits every bot ancestor in a reply chain,
not just the direct parent, so a reply three hops deep in a bot-started
thread still counts toward that thread's root message."""

from unittest.mock import AsyncMock, patch

from tests.builders import make_message_row

from src.feedback.replies import track_reply

BOT_ID = 999
GET_CHAIN = "src.feedback.replies.unified_messages.get_chain"
ADD_REPLY = "src.feedback.replies.message_feedback.add_reply"


class TestTrackReply:
    async def test_direct_reply_to_bot_message_credited_at_depth_one(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()) as add_reply:
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)
        add_reply.assert_awaited_once_with(chat_id=1000, message_id=1, depth=1)

    async def test_reply_two_hops_below_bot_message_credited_at_depth_two(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya"),
            make_message_row(message_id=3, user_id=42, username="vasya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()) as add_reply:
            await track_reply(chat_id=1000, message_id=3, bot_id=BOT_ID)
        add_reply.assert_awaited_once_with(chat_id=1000, message_id=1, depth=2)

    async def test_two_bot_ancestors_both_credited(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya"),
            make_message_row(message_id=3, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=4, user_id=42, username="vasya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()) as add_reply:
            await track_reply(chat_id=1000, message_id=4, bot_id=BOT_ID)
        assert add_reply.await_args_list == [
            ((), {"chat_id": 1000, "message_id": 3, "depth": 1}),
            ((), {"chat_id": 1000, "message_id": 1, "depth": 3}),
        ]

    async def test_reply_with_no_bot_ancestor_credits_nothing(self):
        chain = [
            make_message_row(message_id=1, user_id=42, username="vasya"),
            make_message_row(message_id=2, user_id=43, username="petya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()) as add_reply:
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)
        add_reply.assert_not_awaited()

    async def test_store_failure_does_not_raise(self):
        with patch(GET_CHAIN, AsyncMock(side_effect=RuntimeError("db down"))):
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)  # must not raise
