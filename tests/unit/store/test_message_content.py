"""SQL and enum contract for bot_message_content.

No database in the unit suite, so these pin the statements each function
sends and that the enums carry exactly the values the table's CHECK
constraints allow.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline.state import ContentKind, WorkerStatus
from src.store import message_content

ACQUIRE_TARGET = "src.store.message_content.database.acquire"


def fake_acquire(connection):
    @asynccontextmanager
    async def acquire():
        yield connection

    return acquire


def make_connection(execute_result: str = "INSERT 0 1") -> MagicMock:
    connection = MagicMock()
    connection.execute = AsyncMock(return_value=execute_result)
    return connection


class TestRegister:
    async def test_register_inserts_idempotently(self):
        connection = make_connection()
        with patch(ACQUIRE_TARGET, fake_acquire(connection)):
            await message_content.register(chat_id=1, message_id=2, content_kind="response", worker_status="ok")
        sql, *args = connection.execute.await_args.args
        assert "INSERT INTO bot_message_content" in sql
        assert "ON CONFLICT (chat_id, message_id) DO NOTHING" in sql
        assert tuple(args) == (1, 2, "response", "ok")

    async def test_register_accepts_null_worker_status(self):
        connection = make_connection()
        with patch(ACQUIRE_TARGET, fake_acquire(connection)):
            await message_content.register(chat_id=1, message_id=2, content_kind="error", worker_status=None)
        assert connection.execute.await_args.args[-1] is None


class TestSetResponseLabel:
    async def test_updates_by_primary_key(self):
        connection = make_connection("UPDATE 1")
        with patch(ACQUIRE_TARGET, fake_acquire(connection)):
            await message_content.set_response_label(chat_id=1, message_id=2, label="REFUSED")
        sql, *args = connection.execute.await_args.args
        assert "SET response_label = $3" in sql
        assert "WHERE chat_id = $1 AND message_id = $2" in sql
        assert tuple(args) == (1, 2, "REFUSED")


class TestCleanupOld:
    async def test_returns_deleted_count(self):
        connection = make_connection("DELETE 7")
        with patch(ACQUIRE_TARGET, fake_acquire(connection)):
            deleted = await message_content.cleanup_old()
        assert deleted == 7
        assert "DELETE FROM bot_message_content" in connection.execute.await_args.args[0]

    def test_retention_is_sixty_days(self):
        assert message_content.RETENTION_DAYS == 60


class TestEnums:
    def test_content_kinds_match_the_spec(self):
        assert {kind.value for kind in ContentKind} == {
            "response", "link_summary", "wind_down_reply", "insult_reply",
            "prompt_injection_reply", "limit_notice", "context_limit_notice",
            "cap_notice", "media_failure_notice", "error",
        }

    def test_worker_statuses_match_the_spec(self):
        assert {status.value for status in WorkerStatus} == {"ok", "skipped", "context_overflow", "failed"}
