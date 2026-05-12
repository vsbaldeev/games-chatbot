asyncpg data access layer — connection pool shared across all store modules.

All modules call `async with database.acquire() as conn:` (src/store/db.py) to borrow a
connection from the pool. Pool is initialised at startup via `db.init()` and closed on shutdown.

## Connection pool

```python
pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)

@asynccontextmanager
async def acquire() -> AsyncGenerator[asyncpg.Connection, None]:
    async with pool.acquire() as conn:
        yield conn
```

## Tables

```sql
-- Per-message store; source of truth for reply-chain resolution
unified_messages (
    message_id      BIGINT,
    chat_id         BIGINT,
    user_id         BIGINT,
    username        TEXT,
    content         TEXT,           -- text / transcript / vision description; placeholder before enrichment
    media_type      TEXT,           -- "text" | "voice" | "video_note" | "video" | "photo"
    reply_to_msg_id BIGINT,
    file_id         TEXT,           -- Telegram file_id; permanent; used for lazy photo description
    created_at      DOUBLE PRECISION,
    PRIMARY KEY (chat_id, message_id)
)
INDEX idx_unified_messages_chat_time ON (chat_id, created_at DESC)

-- LLM-extracted facts per user per chat; cap 30 rows per (chat_id, user_id)
user_memories (
    id         BIGSERIAL        PRIMARY KEY,
    chat_id    BIGINT,
    user_id    BIGINT,
    username   TEXT,
    fact       TEXT,
    embedding  vector(384),     -- fastembed MiniLM-L12; NULL until computed
    updated_at DOUBLE PRECISION
)
INDEX idx_user_memories_lookup ON (chat_id, user_id)
INDEX idx_user_memories_hnsw   ON (embedding vector_cosine_ops) WHERE embedding IS NOT NULL  -- HNSW

-- Cumulative counters and max-value stats
user_stats (
    user_id, chat_id, username,
    -- counters incremented via increment_stat()
    sticker_messages, forwarded_messages, voice_messages, video_messages,
    photo_messages, night_messages, animation_messages, duel_wins, roasted_count,
    -- max-value trackers updated via update_max_stat()
    voice_max_duration, long_message_max,
    last_seen INTEGER,
    PRIMARY KEY (user_id, chat_id)
)

-- Deduplication log — each achievement key fires exactly once per user per chat
announced_achievements (
    user_id BIGINT,
    chat_id BIGINT,
    key     TEXT,
    PRIMARY KEY (user_id, chat_id, key)
)

-- Tracks registered chat members for job targeting and achievement queries
chat_members (chat_id BIGINT, user_id BIGINT, username TEXT, PRIMARY KEY (chat_id, user_id))

-- Per-thread conversation history for the response LLM; keyed by reply-chain root or chat_id
thread_history (
    thread_id  TEXT             NOT NULL,
    chat_id    BIGINT           NOT NULL,
    role       TEXT             NOT NULL,   -- "human" | "ai"
    content    TEXT             NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
)
INDEX idx_thread_history_lookup ON (thread_id, created_at)
Retention: 60 days (cleanup_messages_job)
```

## Modules

```
db.py               asyncpg Pool; acquire() context manager; init() / close() lifecycle
unified_messages.py insert (ON CONFLICT DO NOTHING), update_content, get_by_id, get_chain (max 10 hops), get_recent (last N), get_media_group, get_user_messages, cleanup_old
user_memories.py    upsert_facts (cap 30 per user), upsert_hack_attempt, get_facts, get_facts_for_users, get_facts_with_embeddings, find_similar_fact, refresh_updated_at
thread_history.py   append_turn, get_history (thread-scoped, oldest-first), cleanup_old (60-day retention)
embedder.py         embed(text) — fastembed MiniLM-L12 ONNX, returns list[float] (384-dim)
```
