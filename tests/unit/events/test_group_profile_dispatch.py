"""Group-profile-request delivery tests.

Mirrors test_meme_dispatch.py: covers the two units the on-request
group-profile path adds to the events layer:

  - deliver_group_profile: typing indicator -> generate -> honest fallback line
  - run_pipeline: the media-only dispatch branch, sharing the same
                  empty-response-plus-flag pattern the meme path established
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.config.prompts import GROUP_PROFILE_FAILED_REPLIES
from src.events import messages
from tests.builders import make_incoming, make_state

CHAT_ID = 1000
REQUEST_MSG_ID = 55
RUBRIC = "раздай всем роли из Людей Икс"

RUN_GROUP_PROFILE = "src.events.messages.run_group_profile"
SEND_AND_STORE = "src.events.messages.send_and_store"


def make_bot() -> MagicMock:
    bot = MagicMock()
    bot.id = 999
    bot.send_chat_action = AsyncMock()
    return bot


# ---------------------------------------------------------------------------
# deliver_group_profile
# ---------------------------------------------------------------------------

class TestDeliverGroupProfile:
    async def test_sends_the_generated_message_anchored_to_the_request(self):
        bot = make_bot()
        with patch(RUN_GROUP_PROFILE, AsyncMock(return_value="@alice — Циклоп")) as run, \
             patch(SEND_AND_STORE, AsyncMock()) as send:
            await messages.deliver_group_profile(bot, CHAT_ID, REQUEST_MSG_ID, RUBRIC)
        run.assert_awaited_once_with(CHAT_ID, RUBRIC)
        send.assert_awaited_once_with(bot, CHAT_ID, "@alice — Циклоп", reply_to=REQUEST_MSG_ID, is_broadcast=True)

    async def test_shows_typing_indicator_before_generating(self):
        bot = make_bot()
        manager = Mock()
        manager.attach_mock(bot.send_chat_action, "action")
        with patch(RUN_GROUP_PROFILE, AsyncMock(return_value="text")) as run, \
             patch(SEND_AND_STORE, AsyncMock()):
            manager.attach_mock(run, "run_group_profile")
            await messages.deliver_group_profile(bot, CHAT_ID, REQUEST_MSG_ID, RUBRIC)
        bot.send_chat_action.assert_awaited_once_with(chat_id=CHAT_ID, action="typing")
        call_names = [call[0] for call in manager.mock_calls]
        assert call_names.index("action") < call_names.index("run_group_profile")

    async def test_no_result_sends_an_honest_line(self):
        bot = make_bot()
        with patch(RUN_GROUP_PROFILE, AsyncMock(return_value=None)), \
             patch(SEND_AND_STORE, AsyncMock()) as send:
            await messages.deliver_group_profile(bot, CHAT_ID, REQUEST_MSG_ID, RUBRIC)
        send.assert_awaited_once()
        assert send.await_args.args[2] in GROUP_PROFILE_FAILED_REPLIES
        assert send.await_args.kwargs["reply_to"] == REQUEST_MSG_ID

    async def test_generation_error_sends_an_honest_line(self):
        bot = make_bot()
        with patch(RUN_GROUP_PROFILE, AsyncMock(side_effect=RuntimeError("groq down"))), \
             patch(SEND_AND_STORE, AsyncMock()) as send:
            await messages.deliver_group_profile(bot, CHAT_ID, REQUEST_MSG_ID, RUBRIC)
        assert send.await_args.args[2] in GROUP_PROFILE_FAILED_REPLIES

    async def test_failed_indicator_does_not_block_generation(self):
        bot = make_bot()
        bot.send_chat_action = AsyncMock(side_effect=RuntimeError("telegram down"))
        with patch(RUN_GROUP_PROFILE, AsyncMock(return_value="text")) as run, \
             patch(SEND_AND_STORE, AsyncMock()):
            await messages.deliver_group_profile(bot, CHAT_ID, REQUEST_MSG_ID, RUBRIC)
        run.assert_awaited_once()


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
    async def test_empty_response_with_group_profile_request_launches_the_task(self):
        update = make_update()
        state = make_state(
            make_incoming(raw_text=RUBRIC), response="", group_profile_request=True
        )
        with patch.object(messages, "launch_group_profile_task") as launch:
            handled, emit = await run_with_final_state(state, update)
        assert handled is True
        launch.assert_called_once()
        assert launch.call_args.args[1:] == (CHAT_ID, REQUEST_MSG_ID, RUBRIC)
        assert emit.call_args.args[1] == "profile"

    async def test_empty_response_without_group_profile_request_stays_ignored(self):
        update = make_update()
        state = make_state(make_incoming(), response="")
        with patch.object(messages, "launch_group_profile_task") as launch:
            handled, emit = await run_with_final_state(state, update)
        assert handled is False
        launch.assert_not_called()
        assert emit.call_args.args[1] == "ignored"

    @pytest.mark.parametrize("response_text", ["не сегодня", "  ждём  "])
    async def test_text_response_never_takes_the_media_only_branch(self, response_text):
        """A wound-down refusal is text, so the profile branch must not also fire."""
        update = make_update()
        state = make_state(
            make_incoming(), response=response_text, wind_down=True,
            filter_verdict="GROUP_PROFILE_REQUEST",
        )
        with patch.object(messages, "launch_group_profile_task") as launch, \
             patch.object(messages, "deliver_and_record", AsyncMock()):
            handled, _ = await run_with_final_state(state, update)
        assert handled is True
        launch.assert_not_called()

    async def test_meme_and_group_profile_branches_do_not_collide(self):
        """Both share the empty-response dispatch site; only one flag should fire its task."""
        update = make_update()
        state = make_state(make_incoming(), response="", meme_request=True)
        with patch.object(messages, "launch_meme_task") as launch_meme, \
             patch.object(messages, "launch_group_profile_task") as launch_profile:
            handled, emit = await run_with_final_state(state, update)
        assert handled is True
        launch_meme.assert_called_once()
        launch_profile.assert_not_called()
        assert emit.call_args.args[1] == "meme"


class TestGroupProfileIsBroadcast:
    """The group profile judges every member at once — replies to it are chat among members."""

    async def test_successful_profile_is_marked_broadcast(self):
        bot = make_bot()
        with patch(
            RUN_GROUP_PROFILE,
            new_callable=AsyncMock, return_value="@a — 5/10\n@b — 7/10",
        ), patch(
            SEND_AND_STORE, new_callable=AsyncMock
        ) as mock_send:
            await messages.deliver_group_profile(bot, CHAT_ID, REQUEST_MSG_ID, RUBRIC)
        assert mock_send.await_args.kwargs["is_broadcast"] is True

    async def test_failure_reply_is_not_a_broadcast(self):
        """The canned «не смог» line is an ordinary reply to the asker."""
        bot = make_bot()
        with patch(
            RUN_GROUP_PROFILE,
            new_callable=AsyncMock, return_value=None,
        ), patch(
            SEND_AND_STORE, new_callable=AsyncMock
        ) as mock_send:
            await messages.deliver_group_profile(bot, CHAT_ID, REQUEST_MSG_ID, RUBRIC)
        assert mock_send.await_args.kwargs.get("is_broadcast", False) is False
