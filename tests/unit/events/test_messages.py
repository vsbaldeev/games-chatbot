"""Delivery tests — social-link video-before-text behavior in deliver_response.

Scoped to this one new branch; the rest of deliver_response's existing
behavior (voice replies, jokes, search-notification edits) is untouched and
not re-tested here.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.events.messages import deliver_response, passive_voice_extract
from tests.builders import make_incoming, make_state

REPLY_VOICE_PATCH_TARGET = "src.events.messages.try_send_voice_reply"
TRANSCRIBE_VOICE_PATCH_TARGET = "src.events.messages.transcribe_voice"
UPDATE_CONTENT_PATCH_TARGET = "src.events.messages.unified_messages.update_content"
EXTRACT_AND_SAVE_PATCH_TARGET = "src.events.messages.extract_and_save"


def make_msg() -> MagicMock:
    msg = MagicMock()
    msg.chat_id = 1000
    msg.message_id = 55
    msg.chat.send_action = AsyncMock()
    msg.reply_video = AsyncMock()
    sent = MagicMock(message_id=999)
    msg.reply_text = AsyncMock(return_value=sent)
    return msg


class TestSocialLinkVideoDelivery:
    async def test_video_present_is_sent_before_text_reply(self):
        msg = make_msg()
        manager = Mock()
        manager.attach_mock(msg.reply_video, "reply_video")
        manager.attach_mock(msg.reply_text, "reply_text")
        incoming = make_incoming(media_type="text")
        state = make_state(
            incoming, response_trigger="social_link", social_link_video=b"video bytes",
        )
        with patch(REPLY_VOICE_PATCH_TARGET, new=AsyncMock(return_value=None)):
            await deliver_response(state, msg, "reply text")
        msg.reply_video.assert_awaited_once()
        msg.reply_text.assert_awaited_once_with("reply text")
        call_names = [call[0] for call in manager.mock_calls]
        assert call_names.index("reply_video") < call_names.index("reply_text")

    async def test_no_video_skips_reply_video(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text")
        state = make_state(incoming, response_trigger="social_link", social_link_video=None)
        with patch(REPLY_VOICE_PATCH_TARGET, new=AsyncMock(return_value=None)):
            await deliver_response(state, msg, "reply text")
        msg.reply_video.assert_not_awaited()

    async def test_video_send_failure_still_sends_text_reply(self):
        msg = make_msg()
        msg.reply_video = AsyncMock(side_effect=RuntimeError("upload failed"))
        incoming = make_incoming(media_type="text")
        state = make_state(
            incoming, response_trigger="social_link", social_link_video=b"video bytes",
        )
        with patch(REPLY_VOICE_PATCH_TARGET, new=AsyncMock(return_value=None)):
            await deliver_response(state, msg, "reply text")
        msg.reply_video.assert_awaited_once()
        msg.reply_text.assert_awaited_once_with("reply text")


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


class TestShortsVideoDelivery:
    async def test_video_present_is_sent_before_text_reply(self):
        msg = make_msg()
        manager = Mock()
        manager.attach_mock(msg.reply_video, "reply_video")
        manager.attach_mock(msg.reply_text, "reply_text")
        incoming = make_incoming(media_type="text")
        state = make_state(
            incoming, response_trigger="youtube_short", youtube_short_video=b"video bytes",
        )
        with patch(REPLY_VOICE_PATCH_TARGET, new=AsyncMock(return_value=None)):
            await deliver_response(state, msg, "reply text")
        msg.reply_video.assert_awaited_once()
        msg.reply_text.assert_awaited_once_with("reply text")
        call_names = [call[0] for call in manager.mock_calls]
        assert call_names.index("reply_video") < call_names.index("reply_text")

    async def test_no_video_skips_reply_video(self):
        msg = make_msg()
        incoming = make_incoming(media_type="text")
        state = make_state(incoming, response_trigger="youtube_short", youtube_short_video=None)
        with patch(REPLY_VOICE_PATCH_TARGET, new=AsyncMock(return_value=None)):
            await deliver_response(state, msg, "reply text")
        msg.reply_video.assert_not_awaited()

    async def test_video_send_failure_still_sends_text_reply(self):
        msg = make_msg()
        msg.reply_video = AsyncMock(side_effect=RuntimeError("upload failed"))
        incoming = make_incoming(media_type="text")
        state = make_state(
            incoming, response_trigger="youtube_short", youtube_short_video=b"video bytes",
        )
        with patch(REPLY_VOICE_PATCH_TARGET, new=AsyncMock(return_value=None)):
            await deliver_response(state, msg, "reply text")
        msg.reply_video.assert_awaited_once()
        msg.reply_text.assert_awaited_once_with("reply text")
