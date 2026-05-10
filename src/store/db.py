"""Single shared asyncpg connection pool for the whole application."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from pgvector.asyncpg import register_vector

from src import config

pool: asyncpg.Pool | None = None


async def init() -> None:
    global pool
    bootstrap = await asyncpg.connect(config.DATABASE_URL)
    await bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await bootstrap.close()
    pool = await asyncpg.create_pool(
        config.DATABASE_URL, min_size=2, max_size=10, init=register_vector
    )


@asynccontextmanager
async def acquire() -> AsyncGenerator[asyncpg.Connection, None]:
    if pool is None:
        raise RuntimeError("Database not initialized — call db.init() first.")
    async with pool.acquire() as conn:
        yield conn


async def close() -> None:
    global pool
    if pool is not None:
        await pool.close()
        pool = None
