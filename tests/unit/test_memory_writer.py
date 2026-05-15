"""
MemoryWriter tests.

Scenario from c21fe1c: forwarded messages must never trigger fact extraction,
even when the message is long enough or has a bot reply attached.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline.memory_writer import MemoryWriter
from tests.builders import make_incoming, make_state

LONG_TEXT = "это очень длинное сообщение которое превышает минимальный порог для извлечения фактов"


@pytest.fixture
def memory_writer() -> MemoryWriter:
    return MemoryWriter()


class TestForwardedMessages:
    async def test_forwarded_message_never_triggers_extraction(self, memory_writer):
        """Regression (c21fe1c): forwarded content must not become facts about the forwarder."""
        incoming = make_incoming(is_forwarded=True, raw_text=LONG_TEXT)
        state = make_state(incoming, response="bot reply text")

        with patch("src.pipeline.memory_writer.asyncio.create_task") as mock_create_task:
            result = await memory_writer(state)

        mock_create_task.assert_not_called()
        assert result == {}

    async def test_forwarded_message_skipped_even_with_bot_reply(self, memory_writer):
        incoming = make_incoming(is_forwarded=True, raw_text=LONG_TEXT)
        state = make_state(incoming, response="некий развёрнутый ответ бота на переслано")

        with patch("src.pipeline.memory_writer.asyncio.create_task") as mock_create_task:
            await memory_writer(state)

        mock_create_task.assert_not_called()


class TestNonForwardedMessages:
    async def test_non_forwarded_with_reply_triggers_extraction(self, memory_writer):
        incoming = make_incoming(is_forwarded=False, raw_text=LONG_TEXT)
        state = make_state(incoming, response="ответ бота")

        with patch("src.pipeline.memory_writer.asyncio.create_task") as mock_create_task:
            await memory_writer(state)

        mock_create_task.assert_called_once()

    async def test_short_passive_message_skipped(self, memory_writer):
        incoming = make_incoming(is_forwarded=False, raw_text="ок")
        state = make_state(incoming, response="")

        with patch("src.pipeline.memory_writer.asyncio.create_task") as mock_create_task:
            await memory_writer(state)

        mock_create_task.assert_not_called()
