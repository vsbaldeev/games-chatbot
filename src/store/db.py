"""Single shared aiosqlite connection for the whole application."""

import aiosqlite

from src import config

db: aiosqlite.Connection | None = None


async def init() -> None:
    """Open the database connection and configure session-level PRAGMAs."""
    global db
    db = await aiosqlite.connect(config.SQLITE_DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA foreign_keys=ON")


async def get() -> aiosqlite.Connection:
    if db is None:
        raise RuntimeError("Database not initialized — call db.init() first.")
    return db


async def close() -> None:
    global db
    if db is not None:
        await db.close()
        db = None
