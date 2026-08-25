"""Delivery tests — delegation to link_repost and passive voice extraction in
deliver_response.

Scoped to these branches; the rest of deliver_response's existing behavior
(voice replies, jokes, search-notification edits) is untouched and not
re-tested here.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agent import ContextLengthError, DailyLimitError, RateLimitError
from src.events.messages import (
    GENERIC_FAILURE_NOTICE,
    RATE_LIMIT_NOTICE,
    deliver_and_record,
    deliver_response,
    notify_pipeline_failure,
    passive_voice_extract,
    send_limit_notice,
)
from src.events.sending import edit_and_store, send_and_store
from tests.builders import make_incoming, make_state

REPLY_VOICE_PATCH_TARGET = "src.events.messages.try_send_voice_reply"
TRANSCRIBE_VOICE_PATCH_TARGET = "src.events.messages.transcribe_voice"
UPDATE_CONTENT_PATCH_TARGET = "src.events.messages.unified_messages.update_content"
EXTRACT_AND_SAVE_PATCH_TARGET = "src.events.messages.extract_and_save"
LINK_DELIVER_PATCH_TARGET = "src.events.messages.deliver_link_message"
INSERT_PATCH_TARGET = "src.events.messages.unified_messages.insert"
MESSAGES_INSERT_PATCH_TARGET = "src.events.sending.unified_messages.insert"


def make_msg() -> MagicMock:
    msg = MagicMock()
    msg.chat_id = 1000
    msg.message_id = 55
    msg.chat.send_action = AsyncMock()
    sent = MagicMock(message_id=999)
    msg.reply_text = AsyncMock(return_value=sent)
    return msg


class TestLinkTriggerDelegatesToLinkRepost:
    async def test_link_trigger_is_delegated(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text", username="vasya")
        state = make_state(
            incoming, response_trigger="social_link", social_link_content="блок",
            social_link_video=b"video bytes", social_link_url="https://youtu.be/abc",
            link_message_is_bare=True,
        )
        deliver = AsyncMock(return_value=(903, None, "video"))
        with patch(LINK_DELIVER_PATCH_TARGET, new=deliver):
            result = await deliver_response(state, msg, "Про котиков.")
        assert result == (903, None, "video")
        assert deliver.await_args.kwargs == {
            "summary": "Про котиков.", "video": b"video bytes",
            "username": "vasya", "url": "https://youtu.be/abc", "is_bare": True,
            "cap_remaining": None,
        }
        msg.reply_text.assert_not_awaited()

    async def test_cap_remaining_is_forwarded_to_link_repost(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text", username="vasya")
        state = make_state(
            incoming, response_trigger="social_link", social_link_content="блок",
            social_link_video=b"video bytes", social_link_url="https://youtu.be/abc",
            link_message_is_bare=True, social_link_cap_remaining=5,
        )
        deliver = AsyncMock(return_value=(903, None, "video"))
        with patch(LINK_DELIVER_PATCH_TARGET, new=deliver):
            await deliver_response(state, msg, "Про котиков.")
        assert deliver.await_args.kwargs["cap_remaining"] == 5

    async def test_non_bare_message_is_delegated_as_not_bare(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text", username="vasya")
        state = make_state(
            incoming, response_trigger="social_link", social_link_content="блок",
            social_link_video=b"video bytes", social_link_url="https://youtu.be/abc",
            link_message_is_bare=False,
        )
        deliver = AsyncMock(return_value=(901, 55, "video"))
        with patch(LINK_DELIVER_PATCH_TARGET, new=deliver):
            await deliver_response(state, msg, "Про котиков.")
        assert deliver.await_args.kwargs["is_bare"] is False

    async def test_missing_bare_flag_defaults_to_not_bare(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text", username="vasya")
        state = make_state(
            incoming, response_trigger="social_link", social_link_content="блок",
            social_link_video=b"video bytes", social_link_url="https://youtu.be/abc",
        )
        deliver = AsyncMock(return_value=(901, 55, "video"))
        with patch(LINK_DELIVER_PATCH_TARGET, new=deliver):
            await deliver_response(state, msg, "Про котиков.")
        assert deliver.await_args.kwargs["is_bare"] is False

    async def test_failed_fetch_falls_back_to_an_ordinary_reply(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text")
        state = make_state(incoming, response_trigger="social_link")
        deliver = AsyncMock()
        with patch(LINK_DELIVER_PATCH_TARGET, new=deliver), \
             patch(REPLY_VOICE_PATCH_TARGET, new=AsyncMock(return_value=None)):
            await deliver_response(state, msg, "Про котиков.")
        deliver.assert_not_awaited()
        msg.reply_text.assert_awaited_once_with("Про котиков.")


class TestPassiveVoiceExtractPersistsTranscript:
    async def test_voice_transcript_is_persisted(self):
        with patch(TRANSCRIBE_VOICE_PATCH_TARGET, new=AsyncMock(return_value=("привет всем", False))), \
             patch(UPDATE_CONTENT_PATCH_TARGET, new=AsyncMock()) as mock_update, \
             patch(EXTRACT_AND_SAVE_PATCH_TARGET, new=AsyncMock()):
            await passive_voice_extract(
                file_id="file123", media_type="voice", bot=MagicMock(),
                chat_id=1000, user_id=42, username="alice", message_id=555,
            )
        mock_update.assert_awaited_once_with(chat_id=1000, message_id=555, content="привет всем")

    async def test_video_note_transcript_is_not_persisted(self):
        """video_note rows are normally enriched as transcript+frames; writing
        a bare transcript over the placeholder would be a worse, inconsistent
        state, so persistence stays scoped to voice."""
        with patch(TRANSCRIBE_VOICE_PATCH_TARGET, new=AsyncMock(return_value=("привет всем", False))), \
             patch(UPDATE_CONTENT_PATCH_TARGET, new=AsyncMock()) as mock_update, \
             patch(EXTRACT_AND_SAVE_PATCH_TARGET, new=AsyncMock()):
            await passive_voice_extract(
                file_id="file123", media_type="video_note", bot=MagicMock(),
                chat_id=1000, user_id=42, username="alice", message_id=555,
            )
        mock_update.assert_not_awaited()

    async def test_empty_transcript_is_not_persisted(self):
        with patch(TRANSCRIBE_VOICE_PATCH_TARGET, new=AsyncMock(return_value=("", False))), \
             patch(UPDATE_CONTENT_PATCH_TARGET, new=AsyncMock()) as mock_update, \
             patch(EXTRACT_AND_SAVE_PATCH_TARGET, new=AsyncMock()):
            await passive_voice_extract(
                file_id="file123", media_type="voice", bot=MagicMock(),
                chat_id=1000, user_id=42, username="alice", message_id=555,
            )
        mock_update.assert_not_awaited()


class TestDeliverAndRecordPersistsLinkMaterial:
    async def test_shorts_content_is_persisted_as_link_material(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text")
        state = make_state(
            incoming, response_trigger="youtube_short",
            youtube_short_content="[YouTube Shorts] транскрипт и кадры",
        )
        deliver = AsyncMock(return_value=(901, None, "video"))
        with patch(LINK_DELIVER_PATCH_TARGET, new=deliver), \
             patch(INSERT_PATCH_TARGET, new=AsyncMock()) as mock_insert:
            await deliver_and_record(state, msg, bot_id=42, response_text="Про котиков.")
        assert mock_insert.await_args.kwargs["link_material"] == "[YouTube Shorts] транскрипт и кадры"

    async def test_social_link_content_is_persisted_as_link_material(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text")
        state = make_state(
            incoming, response_trigger="social_link",
            social_link_content="[Instagram Reel] подпись и комментарии",
        )
        deliver = AsyncMock(return_value=(901, None, "video"))
        with patch(LINK_DELIVER_PATCH_TARGET, new=deliver), \
             patch(INSERT_PATCH_TARGET, new=AsyncMock()) as mock_insert:
            await deliver_and_record(state, msg, bot_id=42, response_text="Реакция.")
        assert mock_insert.await_args.kwargs["link_material"] == "[Instagram Reel] подпись и комментарии"

    async def test_ordinary_reply_persists_no_link_material(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text")
        state = make_state(incoming, response_trigger="explicit")
        with patch(REPLY_VOICE_PATCH_TARGET, new=AsyncMock(return_value=None)), \
             patch(INSERT_PATCH_TARGET, new=AsyncMock()) as mock_insert:
            await deliver_and_record(state, msg, bot_id=42, response_text="Ответ.")
        assert mock_insert.await_args.kwargs["link_material"] is None


class TestSendAndStoreBroadcastFlag:
    """send_and_store must forward is_broadcast to the store (2026-08-18 addressee gate)."""

    async def test_broadcast_flag_is_forwarded_to_insert(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=MagicMock(message_id=777))
        with patch(
            "src.events.sending.unified_messages.insert", new_callable=AsyncMock
        ) as mock_insert:
            await send_and_store(bot, 1000, "🏷 Роли недели:", is_broadcast=True)
        assert mock_insert.await_args.kwargs["is_broadcast"] is True

    async def test_ordinary_send_defaults_to_not_broadcast(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=MagicMock(message_id=778))
        with patch(
            "src.events.sending.unified_messages.insert", new_callable=AsyncMock
        ) as mock_insert:
            await send_and_store(bot, 1000, "обычный ответ")
        assert mock_insert.await_args.kwargs["is_broadcast"] is False


class TestEditAndStore:
    async def test_edits_message_and_persists_it(self):
        message = MagicMock(message_id=321)
        message.edit_text = AsyncMock()
        with patch(MESSAGES_INSERT_PATCH_TARGET, new_callable=AsyncMock) as mock_insert:
            await edit_and_store(message, 1000, "готово", reply_to=55)
        message.edit_text.assert_awaited_once_with("готово")
        assert mock_insert.await_args.kwargs["message_id"] == 321
        assert mock_insert.await_args.kwargs["content"] == "готово"
        assert mock_insert.await_args.kwargs["reply_to_msg_id"] == 55


def make_failing_msg(message_id: int = 55) -> MagicMock:
    msg = MagicMock()
    msg.message_id = message_id
    msg.get_bot = MagicMock(return_value=MagicMock())
    return msg


def make_notification(message_id: int = 321) -> MagicMock:
    notification = MagicMock(message_id=message_id)
    notification.edit_text = AsyncMock()
    return notification


class TestNotifyPipelineFailureResolvesSearchNotification:
    """A dangling «🔍 Ищу…» message must be edited into the failure notice,
    never left hanging alongside a second, separate message (reported bug:
    the bot was sending both)."""

    async def test_rate_limit_edits_notification_instead_of_sending(self):
        msg = make_failing_msg()
        notification = make_notification()
        with patch(MESSAGES_INSERT_PATCH_TARGET, new_callable=AsyncMock), \
             patch("src.events.messages.send_and_store", new_callable=AsyncMock) as mock_send:
            kind = await notify_pipeline_failure(RateLimitError("x"), msg, 900001, True, notification)
        assert kind == "RateLimit"
        notification.edit_text.assert_awaited_once_with(RATE_LIMIT_NOTICE)
        mock_send.assert_not_awaited()

    async def test_generic_exception_edits_notification_instead_of_sending(self):
        msg = make_failing_msg()
        notification = make_notification()
        with patch(MESSAGES_INSERT_PATCH_TARGET, new_callable=AsyncMock), \
             patch("src.events.messages.send_and_store", new_callable=AsyncMock) as mock_send:
            kind = await notify_pipeline_failure(ValueError("boom"), msg, 900002, True, notification)
        assert kind == "Exception"
        notification.edit_text.assert_awaited_once_with(GENERIC_FAILURE_NOTICE)
        mock_send.assert_not_awaited()

    async def test_context_length_edits_notification_instead_of_sending(self):
        msg = make_failing_msg()
        notification = make_notification()
        with patch(MESSAGES_INSERT_PATCH_TARGET, new_callable=AsyncMock), \
             patch("src.events.messages.send_and_store", new_callable=AsyncMock) as mock_send:
            kind = await notify_pipeline_failure(ContextLengthError("too long"), msg, 900003, True, notification)
        assert kind == "ContextLength"
        notification.edit_text.assert_awaited_once()
        mock_send.assert_not_awaited()

    async def test_daily_limit_edits_notification_instead_of_sending(self):
        msg = make_failing_msg()
        notification = make_notification()
        with patch(MESSAGES_INSERT_PATCH_TARGET, new_callable=AsyncMock), \
             patch("src.events.messages.send_and_store", new_callable=AsyncMock) as mock_send:
            kind = await notify_pipeline_failure(DailyLimitError("done"), msg, 900004, True, notification)
        assert kind == "DailyLimit"
        notification.edit_text.assert_awaited_once()
        mock_send.assert_not_awaited()

    async def test_no_notification_falls_back_to_sending_a_new_message(self):
        msg = make_failing_msg()
        with patch("src.events.messages.send_and_store", new_callable=AsyncMock) as mock_send:
            kind = await notify_pipeline_failure(RateLimitError("x"), msg, 900005, True, None)
        assert kind == "RateLimit"
        mock_send.assert_awaited_once()

    async def test_unaddressed_sends_nothing_even_with_a_notification(self):
        msg = make_failing_msg()
        notification = make_notification()
        with patch("src.events.messages.send_and_store", new_callable=AsyncMock) as mock_send:
            await notify_pipeline_failure(RateLimitError("x"), msg, 900006, False, notification)
        notification.edit_text.assert_not_awaited()
        mock_send.assert_not_awaited()


class TestSendLimitNoticeWithNotification:
    async def test_notification_bypasses_cooldown_reaction_fallback(self):
        """Even mid-cooldown, a stranded search notification gets the full
        text edited in — the reaction-only degrade only applies to a fresh send."""
        msg = make_failing_msg()
        notification = make_notification()
        with patch(MESSAGES_INSERT_PATCH_TARGET, new_callable=AsyncMock):
            await send_limit_notice(msg, 900007, RATE_LIMIT_NOTICE, notification_msg=notification)
            await send_limit_notice(msg, 900007, RATE_LIMIT_NOTICE, notification_msg=notification)
        assert notification.edit_text.await_count == 2
        msg.set_reaction.assert_not_called()
