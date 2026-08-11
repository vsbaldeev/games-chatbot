"""Unit tests for the shared meme sender.

The single place a meme reaches Telegram, used by both the daily job and the
on-request path, so the caption-free contract and the anchoring difference
between the two callers are pinned here.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.memes import sender
from src.store import unified_messages

CHAT_ID = 1000
BOT_ID = 999

GET_MEME = "src.memes.sender.get_meme"
INSERT = "src.memes.sender.unified_messages.insert"


def make_bot() -> MagicMock:
    """Telegram Bot whose send_photo is awaitable and returns a sent message."""
    bot = MagicMock()
    bot.id = BOT_ID
    sent = MagicMock(message_id=555)
    sent.photo = [MagicMock(file_id="file-xyz")]
    bot.send_photo = AsyncMock(return_value=sent)
    return bot


class TestSendMeme:
    async def test_sends_photo_and_records_history(self):
        bot = make_bot()
        with patch(GET_MEME, AsyncMock(return_value=b"IMG")), \
             patch(INSERT, AsyncMock()) as insert:
            assert await sender.send_meme(bot, CHAT_ID) is True

        send_kwargs = bot.send_photo.await_args.kwargs
        assert send_kwargs["chat_id"] == CHAT_ID
        assert send_kwargs["photo"] == b"IMG"

        insert_kwargs = insert.await_args.kwargs
        assert insert_kwargs["chat_id"] == CHAT_ID
        assert insert_kwargs["message_id"] == 555
        assert insert_kwargs["user_id"] == BOT_ID
        assert insert_kwargs["media_type"] == "photo"
        assert insert_kwargs["file_id"] == "file-xyz"

    async def test_never_attaches_a_caption(self):
        """The scraped caption belongs to the source channel, not to the bot."""
        bot = make_bot()
        with patch(GET_MEME, AsyncMock(return_value=b"IMG")), patch(INSERT, AsyncMock()):
            await sender.send_meme(bot, CHAT_ID)
        assert "caption" not in bot.send_photo.await_args.kwargs

    async def test_history_keeps_bare_photo_placeholder(self):
        """A bare placeholder is what the photo-description enricher looks for."""
        bot = make_bot()
        with patch(GET_MEME, AsyncMock(return_value=b"IMG")), \
             patch(INSERT, AsyncMock()) as insert:
            await sender.send_meme(bot, CHAT_ID)
        content = insert.await_args.kwargs["content"]
        assert unified_messages.needs_photo_description(content)
        assert "\n" not in content

    async def test_unanchored_when_no_reply_to(self):
        bot = make_bot()
        with patch(GET_MEME, AsyncMock(return_value=b"IMG")), \
             patch(INSERT, AsyncMock()) as insert:
            await sender.send_meme(bot, CHAT_ID)
        assert bot.send_photo.await_args.kwargs["reply_parameters"] is None
        assert insert.await_args.kwargs["reply_to_msg_id"] is None

    async def test_anchors_to_the_requesting_message(self):
        bot = make_bot()
        with patch(GET_MEME, AsyncMock(return_value=b"IMG")), \
             patch(INSERT, AsyncMock()) as insert:
            await sender.send_meme(bot, CHAT_ID, reply_to=42)
        reply_parameters = bot.send_photo.await_args.kwargs["reply_parameters"]
        assert reply_parameters.message_id == 42
        assert reply_parameters.allow_sending_without_reply is True
        assert insert.await_args.kwargs["reply_to_msg_id"] == 42

    async def test_no_meme_available_returns_false(self):
        bot = make_bot()
        with patch(GET_MEME, AsyncMock(return_value=None)), \
             patch(INSERT, AsyncMock()) as insert:
            assert await sender.send_meme(bot, CHAT_ID) is False
        bot.send_photo.assert_not_awaited()
        insert.assert_not_awaited()

    async def test_send_failure_returns_false_without_raising(self):
        bot = make_bot()
        bot.send_photo = AsyncMock(side_effect=RuntimeError("telegram down"))
        with patch(GET_MEME, AsyncMock(return_value=b"IMG")), \
             patch(INSERT, AsyncMock()) as insert:
            assert await sender.send_meme(bot, CHAT_ID) is False
        insert.assert_not_awaited()
