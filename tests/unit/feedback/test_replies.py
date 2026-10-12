"""Reply-to-bot thread tracking: credits every bot ancestor in a reply chain,
not just the direct parent, so a reply three hops deep in a bot-started
thread still counts toward that thread's root message.

Labelling moved off this path (2026-10-09): track_reply enqueues the reply for
the offline labelling job and makes no model call. Every bot ancestor is still
credited, but exactly one label row is enqueued, for the nearest one — a chain
can pass through several bot messages, and labelling the same reply once per
ancestor would pay for the same model call repeatedly.
"""

from unittest.mock import AsyncMock, patch

from tests.builders import make_message_row

from src.feedback.replies import track_reply

BOT_ID = 999
GET_CHAIN = "src.feedback.replies.unified_messages.get_chain"
ADD_REPLY = "src.feedback.replies.message_feedback.add_reply"
ENQUEUE = "src.feedback.replies.reply_labels.enqueue"


class TestTrackReply:
    async def test_direct_reply_to_bot_message_credited_at_depth_one(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()) as add_reply, \
             patch(ENQUEUE, AsyncMock()):
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)
        add_reply.assert_awaited_once_with(chat_id=1000, message_id=1)

    async def test_reply_two_hops_below_bot_message_credited_at_depth_two(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya"),
            make_message_row(message_id=3, user_id=42, username="vasya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()) as add_reply:
            await track_reply(chat_id=1000, message_id=3, bot_id=BOT_ID)
        add_reply.assert_awaited_once_with(chat_id=1000, message_id=1)

    async def test_two_bot_ancestors_both_credited(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya"),
            make_message_row(message_id=3, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=4, user_id=42, username="vasya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()) as add_reply, \
             patch(ENQUEUE, AsyncMock()):
            await track_reply(chat_id=1000, message_id=4, bot_id=BOT_ID)
        assert add_reply.await_args_list == [
            ((), {"chat_id": 1000, "message_id": 3}),
            ((), {"chat_id": 1000, "message_id": 1}),
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


class TestEnqueueForLabelling:
    async def test_direct_reply_enqueued_for_its_bot_parent(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya", content="бред"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()), \
             patch(ENQUEUE, AsyncMock()) as enqueue:
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)
        enqueue.assert_awaited_once_with(
            chat_id=1000, reply_message_id=2, bot_message_id=1, depth=1, media_type="text",
        )

    async def test_two_bot_ancestors_enqueue_only_the_nearest(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya"),
            make_message_row(message_id=3, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=4, user_id=42, username="vasya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()) as add_reply, \
             patch(ENQUEUE, AsyncMock()) as enqueue:
            await track_reply(chat_id=1000, message_id=4, bot_id=BOT_ID)
        assert add_reply.await_count == 2
        enqueue.assert_awaited_once_with(
            chat_id=1000, reply_message_id=4, bot_message_id=3, depth=1, media_type="text",
        )

    async def test_gif_reply_enqueued_with_its_media_type(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya",
                             media_type="animation", content="[animation]"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()), \
             patch(ENQUEUE, AsyncMock()) as enqueue:
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)
        enqueue.assert_awaited_once_with(
            chat_id=1000, reply_message_id=2, bot_message_id=1, depth=1, media_type="animation",
        )

    async def test_reply_with_no_bot_ancestor_enqueues_nothing(self):
        chain = [
            make_message_row(message_id=1, user_id=42, username="vasya"),
            make_message_row(message_id=2, user_id=43, username="petya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()), \
             patch(ENQUEUE, AsyncMock()) as enqueue:
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)
        enqueue.assert_not_awaited()

    async def test_enqueue_failure_does_not_raise(self):
        chain = [
            make_message_row(message_id=1, user_id=BOT_ID, username="bot"),
            make_message_row(message_id=2, user_id=42, username="vasya"),
        ]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(ADD_REPLY, AsyncMock()), \
             patch(ENQUEUE, AsyncMock(side_effect=RuntimeError("db down"))):
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)  # must not raise
