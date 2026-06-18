"""
user_tags store tests.

The store persists each weekly role plus the reason it was assigned so the bot
can later explain a member's role on request. All database access is mocked
through the ``database.acquire`` context manager — no real connection required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.store import user_tags

ACQUIRE = "src.store.user_tags.database.acquire"

CHAT_ID = 1000


def make_db_conn(fetchrow_return: dict | None = None) -> AsyncMock:
    """Build a mock asyncpg connection with the methods the store uses."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.execute = AsyncMock()
    conn.executemany = AsyncMock()
    return conn


def db_acquire(conn: AsyncMock) -> MagicMock:
    """Wrap a mock connection in an async-context-manager stand-in for acquire()."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestUpsertTags:
    async def test_writes_one_row_per_assignment(self):
        conn = make_db_conn()
        assignments = {
            1: {"tag": "Ночной дозор", "reason": "пишет после полуночи"},
            2: {"tag": "Спидранер", "reason": "проходит игры за день"},
        }
        with patch(ACQUIRE, return_value=db_acquire(conn)):
            await user_tags.upsert_tags(chat_id=CHAT_ID, assignments=assignments)

        conn.executemany.assert_awaited_once()
        rows = conn.executemany.await_args[0][1]
        # Each row carries (chat_id, user_id, tag, reason, assigned_at).
        by_user = {row[1]: row for row in rows}
        assert by_user[1][0] == CHAT_ID
        assert by_user[1][2] == "Ночной дозор"
        assert by_user[1][3] == "пишет после полуночи"
        assert by_user[2][2] == "Спидранер"
        assert len(rows) == 2

    async def test_empty_assignments_does_not_touch_db(self):
        conn = make_db_conn()
        with patch(ACQUIRE, return_value=db_acquire(conn)):
            await user_tags.upsert_tags(chat_id=CHAT_ID, assignments={})
        conn.executemany.assert_not_awaited()

    async def test_upsert_statement_uses_on_conflict(self):
        conn = make_db_conn()
        assignments = {1: {"tag": "Тег", "reason": "причина"}}
        with patch(ACQUIRE, return_value=db_acquire(conn)):
            await user_tags.upsert_tags(chat_id=CHAT_ID, assignments=assignments)
        sql = conn.executemany.await_args[0][0]
        assert "ON CONFLICT" in sql.upper()


class TestGetTag:
    async def test_returns_tag_and_reason_when_row_exists(self):
        conn = make_db_conn(fetchrow_return={"tag": "Ночной дозор", "reason": "пишет ночью"})
        with patch(ACQUIRE, return_value=db_acquire(conn)):
            result = await user_tags.get_tag(chat_id=CHAT_ID, user_id=1)
        assert result == {"tag": "Ночной дозор", "reason": "пишет ночью"}

    async def test_returns_none_when_no_row(self):
        conn = make_db_conn(fetchrow_return=None)
        with patch(ACQUIRE, return_value=db_acquire(conn)):
            result = await user_tags.get_tag(chat_id=CHAT_ID, user_id=999)
        assert result is None
