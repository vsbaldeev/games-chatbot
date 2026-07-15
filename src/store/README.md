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
    media_type      TEXT,           -- "text" | "voice" | "video_note" | "video" | "photo" | "sticker" | "animation" | "audio"
    reply_to_msg_id BIGINT,
    file_id         TEXT,           -- Telegram file_id; permanent; used for lazy photo/sticker description
    media_group_id  TEXT,           -- Telegram album id; groups items of one album
    is_forwarded    BOOLEAN,        -- forwarded channel content, not the sender's own words; rendered as [переслал] in prompts
    created_at      DOUBLE PRECISION,
    PRIMARY KEY (chat_id, message_id)
)
INDEX idx_unified_messages_chat_time ON (chat_id, created_at DESC)

-- LLM-extracted facts per user per chat; cap 30 rows per (chat_id, user_id).
-- Facts untouched for 90 days are deleted by the nightly cleanup
-- (cleanup_stale) — counters included; the dedup path refreshes updated_at
-- on every re-observation, so live facts survive. Cross-user facts carry a
-- «по словам @X, …» attribution prefix in the fact text.
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

-- Per-thread conversation history for the response LLM; keyed by reply-chain
-- root ({chat_id}_{root_message_id}). Flat (non-reply) exchanges are stored
-- under the prospective chain root — the triggering message id — so a
-- follow-up reply chain starts pre-seeded. Legacy chat_id-only rows are dead
-- data aged out by retention.
thread_history (
    thread_id  TEXT             NOT NULL,
    chat_id    BIGINT           NOT NULL,
    role       TEXT             NOT NULL,   -- "human" | "ai"
    content    TEXT             NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
)
INDEX idx_thread_history_lookup ON (thread_id, created_at)
Retention: 60 days (cleanup_messages_job)

-- Жора's own life canon: episode rows are full posted life-story entries
-- (narrative continuity for the next episode, cap 100, oldest pruned);
-- fact rows are short durable canon sentences injected into chat replies
-- (cap 300, semantically deduped on upsert, newest text wins on a match).
-- current_activity/post_format/posted_at are episode-only columns; the
-- newest episode's current_activity is the sole "what is Жора doing right
-- now" answer, decayed by age (fresh/recent/stale) in the context builder.
bot_memories (
    id                BIGSERIAL        PRIMARY KEY,
    kind              TEXT,            -- "episode" | "fact"
    content           TEXT,            -- full episode text, or one fact sentence
    embedding         vector(384),     -- fastembed MiniLM-L12
    post_format       TEXT,            -- episodes only: "story" (more formats later)
    posted_at         DOUBLE PRECISION,-- episodes only: also the catch-up watermark
    current_activity  TEXT,            -- episodes only: present-tense residual
    created_at        DOUBLE PRECISION,
    updated_at        DOUBLE PRECISION
)
INDEX idx_bot_memories_kind_time ON (kind, updated_at DESC)
INDEX idx_bot_memories_hnsw      ON (embedding vector_cosine_ops) WHERE embedding IS NOT NULL

-- One leaky-bucket attention score per (chat, user) for the conversation
-- wind-down engine (src/pipeline/engagement_gate.py). Decay (30-min
-- half-life) is computed lazily in SQL on each access; writing is a single
-- atomic INSERT … ON CONFLICT DO UPDATE … RETURNING, so concurrent messages
-- from the same user serialize on the row lock. Persisted so a redeploy
-- never resets a wound-down user. No retention — dormant rows are tiny and
-- decay to the full-reply tier on next read.
engagement_scores (
    chat_id        BIGINT,
    user_id        BIGINT,
    score          DOUBLE PRECISION,
    last_signal_at DOUBLE PRECISION,
    PRIMARY KEY (chat_id, user_id)
)

-- Vision descriptions per sticker identity. file_unique_id is stable across
-- resends and bots (unlike file_id), so each distinct sticker is described
-- by the vision LLM at most once ever. No retention — rows are tiny and
-- permanently valid.
sticker_descriptions (
    file_unique_id TEXT             PRIMARY KEY,
    description    TEXT             NOT NULL,
    created_at     DOUBLE PRECISION NOT NULL
)
```

## Coverage gaps in `unified_messages`

Some messages are never stored: other bots' posts, rows past the 60-day
retention, and messages sent while the bot was down. Replied-to context is
therefore resolved DB-first with an update fallback — consumers receive a
row-shaped dict synthesized from the Telegram `reply_to_message` object when
the store has no row. The fallback is read-side only and never inserted.
User replies to game messages and the bot's own out-of-pipeline sends
(pipeline error notices) are persisted.

## Modules

```
db.py               asyncpg Pool; acquire() context manager; init() / close() lifecycle
unified_messages.py insert (ON CONFLICT DO NOTHING), update_content, get_by_id, get_chain (max 10 hops), get_recent (last N), get_media_group, get_user_messages, cleanup_old
user_memories.py    upsert_facts (cap 30 per user), upsert_hack_attempt, get_facts, get_facts_for_users, get_facts_with_embeddings, find_similar_fact, refresh_updated_at, cleanup_stale (90-day retention)
bot_memories.py     insert_episode (cap 100, prunes oldest), get_recent_episodes, get_latest_posted_at, get_current_activity, get_facts, get_writer_facts (newest + sampled older), find_similar_facts, find_similar_episodes (similarity floor), upsert_facts (cap 300, semantic dedup, newest wins)
engagement.py       add_signal (atomic decay-and-charge, returns new score), peek_score (read-only decayed score, 0.0 when absent)
thread_history.py   append_turn, get_history (thread-scoped, oldest-first), cleanup_old (60-day retention)
sticker_descriptions.py get_description / save_description — permanent vision-description cache keyed by sticker file_unique_id
embedder.py         embed(text) — fastembed MiniLM-L12 ONNX, returns list[float] (384-dim)
```
