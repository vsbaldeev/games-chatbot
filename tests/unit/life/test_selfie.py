"""Feedback-registration coverage for the chat-requested selfie's final photo send.

The scene-writing/generation/ack paths are untested here — this file covers
only the new message_feedback.register call added on the photo delivery.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.life import selfie

CHAT_ID = 1000
REPLY_TO_MSG_ID = 55

INSERT = "src.life.selfie.unified_messages.insert"
FEEDBACK_REGISTER = "src.life.selfie.message_feedback.register"


def make_bot() -> MagicMock:
    bot = MagicMock()
    sent = MagicMock(message_id=321)
    sent.photo = [MagicMock(file_id="file-abc")]
    bot.send_photo = AsyncMock(return_value=sent)
    return bot


class TestSendAndRecordPhotoRegistersFeedback:
    async def test_registers_with_selfie_source(self):
        bot = make_bot()
        with patch(INSERT, AsyncMock()), \
             patch(FEEDBACK_REGISTER, AsyncMock()) as register:
            await selfie.send_and_record_photo(bot, CHAT_ID, REPLY_TO_MSG_ID, b"PNG")
        assert register.await_args.kwargs == {"chat_id": CHAT_ID, "message_id": 321, "source": "selfie"}

    async def test_feedback_registration_failure_does_not_raise(self):
        bot = make_bot()
        with patch(INSERT, AsyncMock()), \
             patch(FEEDBACK_REGISTER, AsyncMock(side_effect=RuntimeError("db down"))):
            await selfie.send_and_record_photo(bot, CHAT_ID, REPLY_TO_MSG_ID, b"PNG")  # must not raise
