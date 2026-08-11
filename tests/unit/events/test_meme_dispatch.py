"""Meme-request delivery tests.

Covers the two units the on-request meme path adds to the events layer:

  - deliver_meme: typing indicator -> send -> honest fallback line
  - run_pipeline: the media-only dispatch branch, the only path that answers
                  with an image and no text at all

The fallback line carries more weight than usual: the /meme command is gone,
so a fail-closed vision gate must never leave a direct request in silence.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.config.prompts import MEME_FAILED_REPLIES
from src.events import messages
from tests.builders import make_incoming, make_state

CHAT_ID = 1000
REQUEST_MSG_ID = 55

SEND_MEME = "src.events.messages.send_meme"
SEND_AND_STORE = "src.events.messages.send_and_store"


def make_bot() -> MagicMock:
    bot = MagicMock()
    bot.id = 999
    bot.send_chat_action = AsyncMock()
    return bot


# ---------------------------------------------------------------------------
# deliver_meme
# ---------------------------------------------------------------------------

class TestDeliverMeme:
    async def test_sends_meme_anchored_to_the_request(self):
        bot = make_bot()
        with patch(SEND_MEME, AsyncMock(return_value=True)) as send_meme, \
             patch(SEND_AND_STORE, AsyncMock()) as fallback:
            await messages.deliver_meme(bot, CHAT_ID, REQUEST_MSG_ID)
        send_meme.assert_awaited_once_with(bot, CHAT_ID, reply_to=REQUEST_MSG_ID)
        fallback.assert_not_awaited()

    async def test_shows_upload_indicator_before_sending(self):
        """With no text reply, the indicator is the only sign of life."""
        bot = make_bot()
        manager = Mock()
        manager.attach_mock(bot.send_chat_action, "action")
        with patch(SEND_MEME, AsyncMock(return_value=True)) as send_meme:
            manager.attach_mock(send_meme, "send_meme")
            await messages.deliver_meme(bot, CHAT_ID, REQUEST_MSG_ID)
        bot.send_chat_action.assert_awaited_once_with(
            chat_id=CHAT_ID, action="upload_photo"
        )
        call_names = [call[0] for call in manager.mock_calls]
        assert call_names.index("action") < call_names.index("send_meme")

    async def test_no_meme_available_sends_an_honest_line(self):
        bot = make_bot()
        with patch(SEND_MEME, AsyncMock(return_value=False)), \
             patch(SEND_AND_STORE, AsyncMock()) as fallback:
            await messages.deliver_meme(bot, CHAT_ID, REQUEST_MSG_ID)
        fallback.assert_awaited_once()
        assert fallback.await_args.args[2] in MEME_FAILED_REPLIES
        assert fallback.await_args.kwargs["reply_to"] == REQUEST_MSG_ID

    async def test_failed_indicator_does_not_block_the_meme(self):
        bot = make_bot()
        bot.send_chat_action = AsyncMock(side_effect=RuntimeError("telegram down"))
        with patch(SEND_MEME, AsyncMock(return_value=True)) as send_meme, \
             patch(SEND_AND_STORE, AsyncMock()):
            await messages.deliver_meme(bot, CHAT_ID, REQUEST_MSG_ID)
        send_meme.assert_awaited_once()


# ---------------------------------------------------------------------------
# run_pipeline dispatch
# ---------------------------------------------------------------------------

def make_update() -> MagicMock:
    update = MagicMock()
    update.update_id = 123456
    update.message.message_id = REQUEST_MSG_ID
    update.effective_chat.id = CHAT_ID
    return update


def pipeline_returning(state: dict):
    """Patch the module PIPELINE so ainvoke resolves to the given final state."""
    pipeline = MagicMock()
    pipeline.ainvoke = AsyncMock(return_value=state)
    return patch.object(messages, "PIPELINE", pipeline)


async def run_with_final_state(final_state: dict, update: MagicMock):
    """Drive run_pipeline with a stubbed graph, returning its boolean result."""
    context = MagicMock()
    context.bot = make_bot()
    with pipeline_returning(final_state), \
         patch.object(messages, "derive_thread_id", AsyncMock(return_value="t")), \
         patch.object(messages, "build_pipeline_state", MagicMock(return_value=dict(final_state))), \
         patch.object(messages, "is_explicitly_addressed", MagicMock(return_value=True)), \
         patch.object(messages, "canonical") as canonical:
        handled = await messages.run_pipeline(update, context, "text", None)
    return handled, canonical.emit


class TestMediaOnlyDispatch:
    async def test_empty_response_with_meme_request_launches_the_send(self):
        update = make_update()
        state = make_state(make_incoming(), response="", meme_request=True)
        with patch.object(messages, "launch_meme_task") as launch:
            handled, emit = await run_with_final_state(state, update)
        assert handled is True
        launch.assert_called_once()
        assert launch.call_args.args[1:] == (CHAT_ID, REQUEST_MSG_ID)
        assert emit.call_args.args[1] == "meme"

    async def test_empty_response_without_meme_request_stays_ignored(self):
        update = make_update()
        state = make_state(make_incoming(), response="")
        with patch.object(messages, "launch_meme_task") as launch:
            handled, emit = await run_with_final_state(state, update)
        assert handled is False
        launch.assert_not_called()
        assert emit.call_args.args[1] == "ignored"

    @pytest.mark.parametrize("response_text", ["не сегодня", "  ждём  "])
    async def test_text_response_never_takes_the_media_only_branch(self, response_text):
        """A wound-down refusal is text, so the meme branch must not also fire."""
        update = make_update()
        state = make_state(make_incoming(), response=response_text, wind_down=True,
                           filter_verdict="MEME_REQUEST")
        with patch.object(messages, "launch_meme_task") as launch, \
             patch.object(messages, "deliver_and_record", AsyncMock()):
            handled, _ = await run_with_final_state(state, update)
        assert handled is True
        launch.assert_not_called()
