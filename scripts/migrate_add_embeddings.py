"""
One-time migration: add embedding column to user_memories and backfill existing facts.

Run once before deploying the embedding-based dedup feature:
    DATABASE_URL=... python scripts/migrate_add_embeddings.py
"""

import asyncio
import os
import sys

import asyncpg
import numpy as np
from fastembed import TextEmbedding
from pgvector.asyncpg import register_vector

BATCH_SIZE = 100
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH")


async def apply_schema(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute(
        "ALTER TABLE user_memories ADD COLUMN IF NOT EXISTS embedding vector(384)"
    )
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_memories_hnsw
        ON user_memories USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
    """)
    print("Schema updated.")


async def backfill(conn: asyncpg.Connection, model: TextEmbedding) -> None:
    rows = await conn.fetch(
        "SELECT id, fact FROM user_memories WHERE embedding IS NULL ORDER BY id"
    )
    total = len(rows)
    if total == 0:
        print("Nothing to backfill.")
        return
    print(f"Backfilling {total} facts...")
    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        texts = [row["fact"] for row in batch]
        embeddings = list(model.embed(texts))
        await conn.executemany(
            "UPDATE user_memories SET embedding = $1 WHERE id = $2",
            [(np.array(emb), row["id"]) for emb, row in zip(embeddings, batch)],
        )
        done = min(batch_start + BATCH_SIZE, total)
        print(f"  {done}/{total}")
    print("Backfill complete.")


async def run() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        sys.exit(1)
    conn = await asyncpg.connect(database_url)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector(conn)
    model = TextEmbedding(MODEL_NAME, cache_dir=CACHE_DIR)
    await apply_schema(conn)
    await backfill(conn, model)
    await conn.close()


asyncio.run(run())
