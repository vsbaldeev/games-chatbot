"""Delivery tests — delegation to link_repost and passive voice extraction in
deliver_response.

Scoped to these branches; the rest of deliver_response's existing behavior
(voice replies, jokes, search-notification edits) is untouched and not
re-tested here.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.events.messages import deliver_response, passive_voice_extract
from tests.builders import make_incoming, make_state

REPLY_VOICE_PATCH_TARGET = "src.events.messages.try_send_voice_reply"
TRANSCRIBE_VOICE_PATCH_TARGET = "src.events.messages.transcribe_voice"
UPDATE_CONTENT_PATCH_TARGET = "src.events.messages.unified_messages.update_content"
EXTRACT_AND_SAVE_PATCH_TARGET = "src.events.messages.extract_and_save"
LINK_DELIVER_PATCH_TARGET = "src.events.messages.deliver_link_message"


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
        }
        msg.reply_text.assert_not_awaited()

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
