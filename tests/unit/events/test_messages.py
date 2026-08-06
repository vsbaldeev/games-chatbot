"""Delivery tests — social-link video-before-text behavior in deliver_response.

Scoped to this one new branch; the rest of deliver_response's existing
behavior (voice replies, jokes, search-notification edits) is untouched and
not re-tested here.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.events.messages import deliver_response
from tests.builders import make_incoming, make_state

REPLY_VOICE_PATCH_TARGET = "src.events.messages.try_send_voice_reply"


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
