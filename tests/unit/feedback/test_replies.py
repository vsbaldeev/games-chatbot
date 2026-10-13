"""Reply counting: the nearest bot message in the chain gets one counter
bumped per reply, decided by a background labelling task so the serving path
never waits on a model call."""

import asyncio
from unittest.mock import AsyncMock, patch

from tests.builders import make_message_row

from src.feedback.reply_labeller import LabellingCallFailed
from src.feedback.replies import classify_reply, count_reply, nearest_bot_ancestor, track_reply

BOT_ID = 999
GET_CHAIN = "src.feedback.replies.unified_messages.get_chain"
COUNT_REPLY = "src.feedback.replies.count_reply"
GET_ORIGIN = "src.feedback.replies.message_feedback.get_origin"
ADD_REPLY = "src.feedback.replies.message_feedback.add_reply"
LABEL_REPLY = "src.feedback.replies.label_reply"

BOT_MESSAGE = make_message_row(message_id=1, user_id=BOT_ID, username="bot", content="Elden Ring вышел в 2021")


def make_reply(*, content: str = "ну и бред", media_type: str = "text") -> dict:
    return make_message_row(message_id=2, user_id=42, username="vasya", content=content, media_type=media_type)


class TestNearestBotAncestor:
    def test_direct_parent(self):
        assert nearest_bot_ancestor([BOT_MESSAGE], BOT_ID) == BOT_MESSAGE

    def test_picks_the_nearest_of_two(self):
        newer_bot = make_message_row(message_id=3, user_id=BOT_ID, username="bot")
        chain = [BOT_MESSAGE, make_reply(), newer_bot]
        assert nearest_bot_ancestor(chain, BOT_ID) == newer_bot

    def test_none_when_no_bot_message(self):
        assert nearest_bot_ancestor([make_reply()], BOT_ID) is None


class TestTrackReply:
    async def test_spawns_counting_for_the_nearest_bot_message(self):
        chain = [BOT_MESSAGE, make_reply()]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(COUNT_REPLY, AsyncMock()) as counter:
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)
            await asyncio.sleep(0)
        counter.assert_awaited_once_with(chat_id=1000, bot_message=BOT_MESSAGE, reply=chain[-1])

    async def test_no_bot_ancestor_counts_nothing(self):
        chain = [make_message_row(message_id=1, user_id=43), make_reply()]
        with patch(GET_CHAIN, AsyncMock(return_value=chain)), \
             patch(COUNT_REPLY, AsyncMock()) as counter:
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)
            await asyncio.sleep(0)
        counter.assert_not_awaited()

    async def test_empty_chain_counts_nothing(self):
        with patch(GET_CHAIN, AsyncMock(return_value=[])), \
             patch(COUNT_REPLY, AsyncMock()) as counter:
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)
            await asyncio.sleep(0)
        counter.assert_not_awaited()

    async def test_chain_failure_does_not_raise(self):
        with patch(GET_CHAIN, AsyncMock(side_effect=RuntimeError("db down"))):
            await track_reply(chat_id=1000, message_id=2, bot_id=BOT_ID)  # must not raise


class TestClassifyReply:
    async def test_media_reply_is_media_without_a_model_call(self):
        with patch(LABEL_REPLY, AsyncMock()) as labeller:
            kind = await classify_reply(bot_message=BOT_MESSAGE, reply=make_reply(media_type="sticker"), source="pipeline")
        assert kind == "media"
        labeller.assert_not_awaited()

    async def test_blank_text_is_unlabelled_without_a_model_call(self):
        with patch(LABEL_REPLY, AsyncMock()) as labeller:
            kind = await classify_reply(bot_message=BOT_MESSAGE, reply=make_reply(content="   "), source="pipeline")
        assert kind == "unlabelled"
        labeller.assert_not_awaited()

    async def test_model_labels_map_to_kinds(self):
        for label, expected in (("POSITIVE", "positive"), ("NEGATIVE", "negative"), ("NEUTRAL", "neutral")):
            with patch(LABEL_REPLY, AsyncMock(return_value=label)):
                kind = await classify_reply(bot_message=BOT_MESSAGE, reply=make_reply(), source="pipeline")
            assert kind == expected

    async def test_unclear_is_unlabelled(self):
        with patch(LABEL_REPLY, AsyncMock(return_value="UNCLEAR")):
            assert await classify_reply(bot_message=BOT_MESSAGE, reply=make_reply(), source="pipeline") == "unlabelled"

    async def test_unparseable_answer_is_unlabelled(self):
        with patch(LABEL_REPLY, AsyncMock(return_value=None)):
            assert await classify_reply(bot_message=BOT_MESSAGE, reply=make_reply(), source="pipeline") == "unlabelled"

    async def test_failed_call_is_unlabelled(self):
        with patch(LABEL_REPLY, AsyncMock(side_effect=LabellingCallFailed("OPENROUTER_API_KEY unset"))):
            assert await classify_reply(bot_message=BOT_MESSAGE, reply=make_reply(), source="pipeline") == "unlabelled"

    async def test_passes_bot_text_reply_text_and_source_to_the_model(self):
        with patch(LABEL_REPLY, AsyncMock(return_value="POSITIVE")) as labeller:
            await classify_reply(bot_message=BOT_MESSAGE, reply=make_reply(content="топ"), source="roast")
        labeller.assert_awaited_once_with(bot_message="Elden Ring вышел в 2021", reply="топ", source="roast")


class TestCountReply:
    async def test_counts_the_classified_kind(self):
        with patch(GET_ORIGIN, AsyncMock(return_value=("pipeline", "explicit"))), \
             patch(LABEL_REPLY, AsyncMock(return_value="NEGATIVE")), \
             patch(ADD_REPLY, AsyncMock()) as add_reply:
            await count_reply(chat_id=1000, bot_message=BOT_MESSAGE, reply=make_reply())
        add_reply.assert_awaited_once_with(chat_id=1000, message_id=1, kind="negative")

    async def test_excluded_trigger_counts_nothing(self):
        with patch(GET_ORIGIN, AsyncMock(return_value=("pipeline", "social_link"))), \
             patch(LABEL_REPLY, AsyncMock()) as labeller, \
             patch(ADD_REPLY, AsyncMock()) as add_reply:
            await count_reply(chat_id=1000, bot_message=BOT_MESSAGE, reply=make_reply())
        labeller.assert_not_awaited()
        add_reply.assert_not_awaited()

    async def test_untracked_bot_message_counts_nothing(self):
        with patch(GET_ORIGIN, AsyncMock(return_value=None)), \
             patch(LABEL_REPLY, AsyncMock()) as labeller, \
             patch(ADD_REPLY, AsyncMock()) as add_reply:
            await count_reply(chat_id=1000, bot_message=BOT_MESSAGE, reply=make_reply())
        labeller.assert_not_awaited()
        add_reply.assert_not_awaited()

    async def test_store_failure_does_not_raise(self):
        with patch(GET_ORIGIN, AsyncMock(side_effect=RuntimeError("db down"))):
            await count_reply(chat_id=1000, bot_message=BOT_MESSAGE, reply=make_reply())  # must not raise
