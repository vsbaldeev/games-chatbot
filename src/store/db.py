"""Single shared aiosqlite connection for the whole application."""

import aiosqlite

from src import config

_db: aiosqlite.Connection | None = None


async def init() -> None:
    """Open the database connection and configure session-level PRAGMAs."""
    global _db
    _db = await aiosqlite.connect(config.SQLITE_DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA synchronous=NORMAL")
    await _db.execute("PRAGMA busy_timeout=5000")
    await _db.execute("PRAGMA foreign_keys=ON")


async def get() -> aiosqlite.Connection:
    if _db is None:
        raise RuntimeError("Database not initialized — call db.init() first.")
    return _db


async def close() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None
