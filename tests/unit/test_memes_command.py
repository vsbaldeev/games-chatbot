"""Unit tests for the /meme command handler.

All collaborators (get_meme, the Telegram update, unified_messages) are mocked;
these tests verify the handler's control flow and the arguments it forwards.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.commands.fun import meme

CHAT_ID = 1000
NO_MEMES_TEXT = "Мемы закончились — загляни попозже."
SEND_FAILED_TEXT = "Не смог отправить мем — попробуй ещё раз."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_update(*, has_message: bool = True, message_id: int = 100) -> MagicMock:
    """Build a Telegram Update with awaitable reply/send_action methods."""
    update = MagicMock()
    if not has_message:
        update.message = None
        return update
    sent = MagicMock(message_id=999)
    sent.photo = [MagicMock(file_id="file-xyz")]
    update.message.message_id = message_id
    update.message.reply_photo = AsyncMock(return_value=sent)
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    update.effective_chat.id = CHAT_ID
    return update


def make_context(bot_id: int = 555) -> MagicMock:
    context = MagicMock()
    context.bot.id = bot_id
    return context


# ---------------------------------------------------------------------------
# cmd_meme
# ---------------------------------------------------------------------------

class TestCmdMeme:
    async def test_no_message_returns_without_sending(self):
        update = make_update(has_message=False)
        with patch("src.commands.fun.meme.get_meme", AsyncMock()) as get_meme:
            await meme.cmd_meme(update, make_context())
        get_meme.assert_not_called()

    async def test_sends_typing_action_before_fetch(self):
        update = make_update()
        with patch("src.commands.fun.meme.get_meme", AsyncMock(return_value=None)):
            await meme.cmd_meme(update, make_context())
        update.message.chat.send_action.assert_awaited_once_with("upload_photo")

    async def test_no_meme_sends_fallback_text(self):
        update = make_update()
        with patch("src.commands.fun.meme.get_meme", AsyncMock(return_value=None)):
            await meme.cmd_meme(update, make_context())
        update.message.reply_text.assert_awaited_once_with(NO_MEMES_TEXT)
        update.message.reply_photo.assert_not_called()

    async def test_happy_path_sends_photo_and_logs_message(self):
        update = make_update()
        with patch("src.commands.fun.meme.get_meme",
                   AsyncMock(return_value=("https://img/x.jpg", "funny"))), \
             patch("src.commands.fun.meme.unified_messages") as unified:
            unified.insert = AsyncMock()
            unified.format_photo_content = MagicMock(return_value="[photo] funny")
            await meme.cmd_meme(update, make_context())
        update.message.reply_photo.assert_awaited_once_with("https://img/x.jpg", caption="funny")
        unified.insert.assert_awaited_once()
        insert_kwargs = unified.insert.call_args.kwargs
        assert insert_kwargs["chat_id"] == CHAT_ID
        assert insert_kwargs["message_id"] == 999
        assert insert_kwargs["media_type"] == "photo"
        assert insert_kwargs["file_id"] == "file-xyz"
        assert insert_kwargs["reply_to_msg_id"] == 100

    async def test_empty_caption_is_sent_as_none(self):
        update = make_update()
        with patch("src.commands.fun.meme.get_meme",
                   AsyncMock(return_value=("https://img/x.jpg", ""))), \
             patch("src.commands.fun.meme.unified_messages") as unified:
            unified.insert = AsyncMock()
            unified.format_photo_content = MagicMock(return_value="[photo]")
            await meme.cmd_meme(update, make_context())
        update.message.reply_photo.assert_awaited_once_with("https://img/x.jpg", caption=None)

    async def test_send_failure_sends_error_text(self):
        update = make_update()
        update.message.reply_photo = AsyncMock(side_effect=RuntimeError("telegram down"))
        with patch("src.commands.fun.meme.get_meme",
                   AsyncMock(return_value=("https://img/x.jpg", "funny"))), \
             patch("src.commands.fun.meme.unified_messages") as unified:
            unified.format_photo_content = MagicMock(return_value="[photo] funny")
            await meme.cmd_meme(update, make_context())
        update.message.reply_text.assert_awaited_once_with(SEND_FAILED_TEXT)
