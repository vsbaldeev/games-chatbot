"""
ContextBuilder tests.

Scenarios anchored to real bugs and new features:
  4550f56 — reply_chain was never populated; worker got recent_history instead
  8da0de2 — replied-to not resolved when older than recent window;
             current message appeared in its own context
  feature  — meme photo in reply chain enriched via vision LLM so bot can
             explain the image when user replies and asks about it

All store and LLM calls are mocked so no database or network is required.
"""

import contextlib
from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline.context_builder import CHAIN_MSG_CHAR_LIMIT, ContextBuilder
from tests.builders import make_incoming, make_message_row, make_state

STORE_GET_RECENT = "src.pipeline.context_builder.unified_messages.get_recent"
STORE_GET_CHAIN = "src.pipeline.context_builder.unified_messages.get_chain"
STORE_GET_BY_ID = "src.pipeline.context_builder.unified_messages.get_by_id"
STORE_UPDATE_CONTENT = "src.pipeline.context_builder.unified_messages.update_content"
STORE_GET_FACTS_FOR_USERS = "src.pipeline.context_builder.user_memories.get_facts_for_users"
STORE_GET_FACTS = "src.pipeline.context_builder.user_memories.get_facts"
DESCRIBE_PHOTO = "src.pipeline.context_builder.describe_photo"


def patch_store(stack: contextlib.ExitStack, *, recent=None, chain=None, by_id=None):
    """Enter all store patches into an ExitStack. Returns (mock_chain, mock_get_by_id)."""
    stack.enter_context(patch(STORE_GET_RECENT, new_callable=AsyncMock, return_value=recent or []))
    mock_chain = stack.enter_context(patch(STORE_GET_CHAIN, new_callable=AsyncMock, return_value=chain or []))
    mock_get_by_id = stack.enter_context(patch(STORE_GET_BY_ID, new_callable=AsyncMock, return_value=by_id))
    stack.enter_context(patch(STORE_UPDATE_CONTENT, new_callable=AsyncMock))
    stack.enter_context(patch(STORE_GET_FACTS_FOR_USERS, new_callable=AsyncMock, return_value={}))
    stack.enter_context(patch(STORE_GET_FACTS, new_callable=AsyncMock, return_value=[]))
    return mock_chain, mock_get_by_id


@pytest.fixture
def context_builder() -> ContextBuilder:
    return ContextBuilder()


class TestReplyChain:
    """reply_chain must be populated from the store when a reply_to_msg_id is present."""

    async def test_reply_chain_populated_from_store(self, context_builder):
        # Bug (4550f56): ContextBuilder was not calling get_chain, so workers
        # fell back silently to recent_history for thread context.
        chain_rows = [make_message_row(message_id=50), make_message_row(message_id=51)]
        incoming = make_incoming(reply_to_msg_id=51)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            mock_chain, _ = patch_store(stack, chain=chain_rows)
            result = await context_builder(state)

        mock_chain.assert_called_once_with(chat_id=1000, message_id=51)
        assert result["context"]["reply_chain"] == chain_rows

    async def test_reply_chain_empty_when_no_reply(self, context_builder):
        incoming = make_incoming(reply_to_msg_id=None)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            mock_chain, _ = patch_store(stack)
            result = await context_builder(state)

        mock_chain.assert_not_called()
        assert result["context"]["reply_chain"] == []


class TestRepliedToResolution:
    """replied_to must be resolved correctly whether the message is in recent or older."""

    async def test_replied_to_found_in_recent_without_extra_db_call(self, context_builder):
        # Bug (8da0de2): when the replied-to message existed in recent, it was
        # still triggering a redundant get_by_id fetch.
        target_row = make_message_row(message_id=99)
        recent_rows = [make_message_row(message_id=98), target_row]
        incoming = make_incoming(message_id=200, reply_to_msg_id=99)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            _, mock_get_by_id = patch_store(stack, recent=recent_rows)
            result = await context_builder(state)

        mock_get_by_id.assert_not_called()
        assert result["context"]["replied_to"] == target_row

    async def test_replied_to_fetched_by_id_when_not_in_recent(self, context_builder):
        old_row = make_message_row(message_id=10)
        recent_rows = [make_message_row(message_id=98)]
        incoming = make_incoming(message_id=200, reply_to_msg_id=10)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            _, mock_get_by_id = patch_store(stack, recent=recent_rows, by_id=old_row)
            result = await context_builder(state)

        mock_get_by_id.assert_called_once_with(chat_id=1000, message_id=10)
        assert result["context"]["replied_to"] == old_row

    async def test_replied_to_is_none_when_no_reply(self, context_builder):
        incoming = make_incoming(message_id=200, reply_to_msg_id=None)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack)
            result = await context_builder(state)

        assert result["context"]["replied_to"] is None


class TestCurrentMessageExclusion:
    async def test_current_message_excluded_from_recent_history(self, context_builder):
        # Bug (8da0de2): the incoming message itself was included in recent_history,
        # causing the bot to see its own prompt as part of context.
        current_row = make_message_row(message_id=200)
        older_row = make_message_row(message_id=198)
        incoming = make_incoming(message_id=200)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack, recent=[current_row, older_row])
            result = await context_builder(state)

        history_ids = [row["message_id"] for row in result["context"]["recent_history"]]
        assert 200 not in history_ids
        assert 198 in history_ids


class TestPhotoEnrichmentInReplyChain:
    """When the user replies to a meme photo and asks to explain it, the context
    builder must describe the photo via vision LLM before workers run."""

    async def test_placeholder_photo_in_chain_is_described(self, context_builder):
        photo_row = make_message_row(
            message_id=77,
            media_type="photo",
            content="[photo]\nReddit post title",
            file_id="tg-file-abc",
        )
        incoming = make_incoming(reply_to_msg_id=77)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack, chain=[photo_row])
            mock_describe = stack.enter_context(
                patch(DESCRIBE_PHOTO, new_callable=AsyncMock, return_value="Мем с котом в шляпе.")
            )
            result = await context_builder(state)

        mock_describe.assert_called_once()
        enriched_content = result["context"]["reply_chain"][0]["content"]
        assert "Мем с котом в шляпе." in enriched_content
        assert "[photo]" not in enriched_content

    async def test_already_described_photo_not_re_described(self, context_builder):
        photo_row = make_message_row(
            message_id=78,
            media_type="photo",
            content="Уже описанное фото — кот сидит на окне.",
            file_id="tg-file-xyz",
        )
        incoming = make_incoming(reply_to_msg_id=78)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack, chain=[photo_row])
            mock_describe = stack.enter_context(
                patch(DESCRIBE_PHOTO, new_callable=AsyncMock, return_value="whatever")
            )
            result = await context_builder(state)

        mock_describe.assert_not_called()
        assert result["context"]["reply_chain"][0]["content"] == photo_row["content"]

    async def test_photo_without_file_id_not_described(self, context_builder):
        photo_row = make_message_row(
            message_id=79,
            media_type="photo",
            content="[photo]",
        )
        incoming = make_incoming(reply_to_msg_id=79)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack, chain=[photo_row])
            mock_describe = stack.enter_context(
                patch(DESCRIBE_PHOTO, new_callable=AsyncMock, return_value="whatever")
            )
            await context_builder(state)

        mock_describe.assert_not_called()

    async def test_description_cached_in_db(self, context_builder):
        photo_row = make_message_row(
            message_id=80,
            media_type="photo",
            content="[photo]",
            file_id="tg-file-cache",
        )
        incoming = make_incoming(reply_to_msg_id=80)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack, chain=[photo_row])
            stack.enter_context(
                patch(DESCRIBE_PHOTO, new_callable=AsyncMock, return_value="Описание мема.")
            )
            mock_update = stack.enter_context(
                patch(STORE_UPDATE_CONTENT, new_callable=AsyncMock)
            )
            await context_builder(state)

        mock_update.assert_called_once_with(chat_id=1000, message_id=80, content="Описание мема.")


class TestChainTruncation:
    """Each message in the reply chain must be capped at CHAIN_MSG_CHAR_LIMIT characters
    before being passed to the worker or response LLM, preventing context-window exhaustion
    from deep threads with verbose bot replies."""

    async def test_short_content_not_truncated(self, context_builder):
        row = make_message_row(message_id=10, content="короткое сообщение")
        incoming = make_incoming(reply_to_msg_id=10)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack, chain=[row])
            result = await context_builder(state)

        assert result["context"]["reply_chain"][0]["content"] == "короткое сообщение"

    async def test_content_at_limit_not_truncated(self, context_builder):
        content = "a" * CHAIN_MSG_CHAR_LIMIT
        row = make_message_row(message_id=11, content=content)
        incoming = make_incoming(reply_to_msg_id=11)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack, chain=[row])
            result = await context_builder(state)

        assert result["context"]["reply_chain"][0]["content"] == content

    async def test_content_over_limit_truncated_with_ellipsis(self, context_builder):
        content = "a" * (CHAIN_MSG_CHAR_LIMIT + 100)
        row = make_message_row(message_id=12, content=content)
        incoming = make_incoming(reply_to_msg_id=12)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack, chain=[row])
            result = await context_builder(state)

        assert result["context"]["reply_chain"][0]["content"] == "a" * CHAIN_MSG_CHAR_LIMIT + "…"

    async def test_all_rows_in_chain_are_independently_truncated(self, context_builder):
        long_content = "x" * (CHAIN_MSG_CHAR_LIMIT + 50)
        rows = [make_message_row(message_id=idx, content=long_content) for idx in range(1, 4)]
        incoming = make_incoming(reply_to_msg_id=3)
        state = make_state(incoming)

        with contextlib.ExitStack() as stack:
            patch_store(stack, chain=rows)
            result = await context_builder(state)

        for chain_row in result["context"]["reply_chain"]:
            assert chain_row["content"] == "x" * CHAIN_MSG_CHAR_LIMIT + "…"
