"""Single shared asyncpg connection pool for the whole application."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from pgvector.asyncpg import register_vector

from src import config

pool: asyncpg.Pool | None = None


async def init() -> None:
    """Create the shared connection pool.

    The ``vector`` extension and all tables are provisioned by Alembic
    migrations (``alembic upgrade head``) before the process starts, so the
    pool's ``register_vector`` codec has a type to bind to.
    """
    global pool
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
