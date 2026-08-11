"""Daily meme job tests.

The job is now a thin fan-out over the shared sender (``src/memes/sender.py``),
so these cover only the fan-out and its failure isolation; what a single send
does is pinned in tests/unit/memes/test_sender.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.jobs import meme

SEND_MEME = "src.jobs.meme.send_meme"
GET_ALL_CHATS = "src.jobs.meme.achievements.get_all_chat_ids"


def make_context() -> MagicMock:
    """Telegram context carrying a bot instance."""
    context = MagicMock()
    context.bot.id = 999
    return context


class TestDailyMemeJob:
    async def test_sends_to_every_chat(self):
        context = make_context()
        with patch(GET_ALL_CHATS, AsyncMock(return_value=[1, 2, 3])), \
             patch(SEND_MEME, AsyncMock(return_value=True)) as send_meme:
            await meme.daily_meme_job(context)

        assert send_meme.await_count == 3
        assert {call.args[1] for call in send_meme.await_args_list} == {1, 2, 3}

    async def test_daily_meme_is_unanchored(self):
        """The daily meme opens the day; it does not reply to anyone."""
        context = make_context()
        with patch(GET_ALL_CHATS, AsyncMock(return_value=[1])), \
             patch(SEND_MEME, AsyncMock(return_value=True)) as send_meme:
            await meme.daily_meme_job(context)

        assert "reply_to" not in send_meme.await_args.kwargs

    async def test_no_chats_does_nothing(self):
        context = make_context()
        with patch(GET_ALL_CHATS, AsyncMock(return_value=[])), \
             patch(SEND_MEME, AsyncMock()) as send_meme:
            await meme.daily_meme_job(context)

        send_meme.assert_not_awaited()

    async def test_one_chat_failure_does_not_abort_others(self):
        context = make_context()
        with patch(GET_ALL_CHATS, AsyncMock(return_value=[1, 2])), \
             patch(SEND_MEME, AsyncMock(side_effect=[RuntimeError("boom"), True])) as send_meme:
            # gather(return_exceptions=True) must keep the second call alive.
            await meme.daily_meme_job(context)

        assert send_meme.await_count == 2
