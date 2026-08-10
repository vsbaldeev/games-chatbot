"""
Daily meme job tests.

Covers two units in isolation, with the fetcher, store, and Telegram bot all
mocked:

  - send_meme_to_chat: fetch vetted bytes -> send_photo -> record in history,
                       plus the skip paths (nothing vetted, send raised)
  - daily_meme_job   : fan-out over every chat returned by get_all_chat_ids
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.jobs import meme
from src.store import unified_messages

CHAT_ID = 1000
BOT_ID = 999

GET_MEME = "src.jobs.meme.get_meme"
INSERT = "src.jobs.meme.unified_messages.insert"
GET_ALL_CHATS = "src.jobs.meme.achievements.get_all_chat_ids"


def make_context() -> MagicMock:
    """Telegram context whose bot exposes an awaitable send_photo and an id."""
    context = MagicMock()
    context.bot.id = BOT_ID
    sent = MagicMock()
    sent.message_id = 555
    sent.photo = [MagicMock(file_id="file-xyz")]
    context.bot.send_photo = AsyncMock(return_value=sent)
    return context


class TestSendMemeToChat:
    async def test_sends_meme_and_records_history(self):
        context = make_context()
        with patch(GET_MEME, new_callable=AsyncMock, return_value=b"bytes"), \
             patch(INSERT, new_callable=AsyncMock) as mock_insert:
            await meme.send_meme_to_chat(context, CHAT_ID)

        context.bot.send_photo.assert_awaited_once()
        send_kwargs = context.bot.send_photo.await_args.kwargs
        assert send_kwargs["chat_id"] == CHAT_ID
        assert send_kwargs["photo"] == b"bytes"

        insert_kwargs = mock_insert.await_args.kwargs
        assert insert_kwargs["chat_id"] == CHAT_ID
        assert insert_kwargs["message_id"] == 555
        assert insert_kwargs["user_id"] == BOT_ID
        assert insert_kwargs["media_type"] == "photo"
        assert insert_kwargs["file_id"] == "file-xyz"

    async def test_never_sends_a_caption(self):
        """Source captions belong to the scraped channel, not to the bot."""
        context = make_context()
        with patch(GET_MEME, new_callable=AsyncMock, return_value=b"bytes"), \
             patch(INSERT, new_callable=AsyncMock):
            await meme.send_meme_to_chat(context, CHAT_ID)

        assert "caption" not in context.bot.send_photo.await_args.kwargs

    async def test_history_keeps_bare_photo_placeholder(self):
        """A bare placeholder is what the photo-description enricher looks for."""
        context = make_context()
        with patch(GET_MEME, new_callable=AsyncMock, return_value=b"bytes"), \
             patch(INSERT, new_callable=AsyncMock) as mock_insert:
            await meme.send_meme_to_chat(context, CHAT_ID)

        content = mock_insert.await_args.kwargs["content"]
        assert unified_messages.needs_photo_description(content)
        assert "\n" not in content

    async def test_no_meme_available_sends_nothing(self):
        context = make_context()
        with patch(GET_MEME, new_callable=AsyncMock, return_value=None), \
             patch(INSERT, new_callable=AsyncMock) as mock_insert:
            await meme.send_meme_to_chat(context, CHAT_ID)

        context.bot.send_photo.assert_not_awaited()
        mock_insert.assert_not_awaited()

    async def test_send_failure_is_swallowed(self):
        context = make_context()
        context.bot.send_photo = AsyncMock(side_effect=RuntimeError("telegram down"))
        with patch(GET_MEME, new_callable=AsyncMock, return_value=b"bytes"), \
             patch(INSERT, new_callable=AsyncMock):
            # Must not raise.
            await meme.send_meme_to_chat(context, CHAT_ID)


class TestDailyMemeJob:
    async def test_sends_to_every_chat(self):
        context = make_context()
        with patch(GET_ALL_CHATS, new_callable=AsyncMock, return_value=[1, 2, 3]), \
             patch("src.jobs.meme.send_meme_to_chat", new_callable=AsyncMock) as mock_send:
            await meme.daily_meme_job(context)

        assert mock_send.await_count == 3
        sent_chat_ids = {call.args[1] for call in mock_send.await_args_list}
        assert sent_chat_ids == {1, 2, 3}

    async def test_no_chats_does_nothing(self):
        context = make_context()
        with patch(GET_ALL_CHATS, new_callable=AsyncMock, return_value=[]), \
             patch("src.jobs.meme.send_meme_to_chat", new_callable=AsyncMock) as mock_send:
            await meme.daily_meme_job(context)

        mock_send.assert_not_awaited()

    async def test_one_chat_failure_does_not_abort_others(self):
        context = make_context()
        with patch(GET_ALL_CHATS, new_callable=AsyncMock, return_value=[1, 2]), \
             patch("src.jobs.meme.send_meme_to_chat", new_callable=AsyncMock,
                   side_effect=[RuntimeError("boom"), None]) as mock_send:
            # gather(return_exceptions=True) must keep the second call alive.
            await meme.daily_meme_job(context)

        assert mock_send.await_count == 2
