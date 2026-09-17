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
    link_material   TEXT,           -- persisted ingestion content block for the bot's own link-repost messages (Shorts/Reel/YouTube); marks rows for the router's addressing gate, re-injected into replies for grounding
    is_broadcast    BOOLEAN NOT NULL DEFAULT FALSE,  -- bot's group-wide announcements (weekly roles, group profile); marks rows for the router's addressing gate
    created_at      DOUBLE PRECISION,
    PRIMARY KEY (chat_id, message_id)
)
INDEX idx_unified_messages_chat_time ON (chat_id, created_at DESC)

-- LLM-extracted facts per user per chat; cap 30 rows per (chat_id, user_id).
-- Facts untouched for 14 days are deleted by the nightly cleanup
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
    photo_messages, night_messages, animation_messages, duel_wins,
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

-- RETIRED 2026-08-10 (Жора's scheduled life posts, see src/life/README.md).
-- The bot_memories table still exists in the database: no migration drops it,
-- so the canon survives if the feature is ever revived. Nothing reads or
-- writes it any more — src/store/bot_memories.py was deleted.

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

-- Deduplicated response system prompt text, keyed by sha256. Stored once so
-- bot_llm_log rows below need only a hash, not the multi-KB prompt itself.
-- Retention: pruned when no bot_llm_log row references it any more.
llm_system_prompts (
    sha256        TEXT        PRIMARY KEY,
    content       TEXT        NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
)

-- One row per delivered pipeline reply that went through the response LLM —
-- the exact prompt and response, for inspecting corrected replies. Only the
-- response call is logged, not the worker/tool-calling call. See
-- src/feedback/README.md. Retention: 60 days.
bot_llm_log (
    chat_id            BIGINT      NOT NULL,
    bot_message_id     BIGINT      NOT NULL,
    user_message_id    BIGINT      NOT NULL,
    user_id            BIGINT      NOT NULL,
    trigger            TEXT        NOT NULL,
    filter_verdict     TEXT        NOT NULL DEFAULT '-',
    system_prompt_sha  TEXT        NOT NULL REFERENCES llm_system_prompts(sha256),
    history_messages   JSONB       NOT NULL,   -- thread-history turns before the final human turn
    user_prompt        TEXT        NOT NULL,   -- final human turn exactly as sent to the model
    raw_response       TEXT        NOT NULL,   -- first model output, before language correction
    language_corrected BOOLEAN     NOT NULL DEFAULT FALSE,
    response           TEXT        NOT NULL,   -- text actually delivered to the chat
    model              TEXT,                   -- model that answered; NULL when unreported
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    latency_ms         INTEGER     NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (chat_id, bot_message_id)
)
INDEX idx_bot_llm_log_created ON (created_at)

-- One row per bot message of any source — reaction/reply/correction counts
-- gathered only within the first 15 minutes after sending (every mutator's
-- WHERE clause enforces the window, except add_correction: the classifier
-- may finish after the window closes but the reply itself arrived in time).
-- See src/feedback/README.md. Retention: 60 days.
bot_message_feedback (
    chat_id           BIGINT      NOT NULL,
    message_id        BIGINT      NOT NULL,
    source            TEXT        NOT NULL,   -- "pipeline" | "meme" | "selfie" | "roles" | "group_profile" | "notice"
    trigger           TEXT        NOT NULL DEFAULT '-',
    filter_verdict    TEXT        NOT NULL DEFAULT '-',
    sent_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    reactions         INTEGER     NOT NULL DEFAULT 0,   -- net current reactions
    emojis            JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- {"🤡": 1, "😂": 2}
    replies           INTEGER     NOT NULL DEFAULT 0,
    corrections       INTEGER     NOT NULL DEFAULT 0,
    thread_depth      INTEGER     NOT NULL DEFAULT 0,
    first_feedback_at TIMESTAMPTZ,
    ignored           BOOLEAN,                -- NULL while the 15-minute window is still open
    PRIMARY KEY (chat_id, message_id)
)
INDEX idx_bot_message_feedback_sent ON (sent_at)

-- Hourly rollup of bot_message_feedback, written by feedback_metrics_job
-- (src/jobs/feedback.py). The only feedback table dashboards should query —
-- counts sit next to their rates so a daily/weekly figure is
-- SUM(count) / SUM(messages), never an average of hourly rates.
-- Retention: 365 days.
bot_feedback_metrics (
    bucket_start             TIMESTAMPTZ NOT NULL,  -- date_trunc('hour', sent_at)
    chat_id                  BIGINT      NOT NULL,
    source                   TEXT        NOT NULL,
    trigger                  TEXT        NOT NULL,
    messages                 INTEGER     NOT NULL,
    ignored                  INTEGER     NOT NULL,
    corrected                INTEGER     NOT NULL,
    reacted                  INTEGER     NOT NULL,
    replied                  INTEGER     NOT NULL,
    ignore_rate              REAL        NOT NULL,
    correction_rate          REAL        NOT NULL,
    reaction_rate            REAL        NOT NULL,
    reply_rate               REAL        NOT NULL,
    avg_thread_depth         REAL        NOT NULL,
    median_first_feedback_s  REAL,
    computed_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bucket_start, chat_id, source, trigger)
)
```

JSONB columns (`history_messages`, `emojis`) are written as `json.dumps(...)`
text with an explicit `$N::jsonb` cast — no JSONB codec is registered on the
connection pool (`db.py` only registers pgvector's), so asyncpg would
otherwise reject a plain Python dict/list for these columns.

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
unified_messages.py insert (ON CONFLICT DO NOTHING, keyword-only: is_broadcast), update_content, get_by_id, get_chain (max 10 hops), get_recent (last N), get_media_group, get_user_messages, cleanup_old
user_memories.py    upsert_facts (cap 30 per user), upsert_hack_attempt, upsert_insult_attempt, is_counter_fact (counter tallies — filtered out of reply prompts), get_facts, get_facts_for_users (whole fact list — weekly roles/engagement), find_relevant_facts_for_users (similarity-gated per-user top-K, used by both context_builder's reply prompts and memory_writer's extraction "existing facts"; ranking, threshold and truncation all in SQL via ROW_NUMBER), get_facts_with_embeddings, find_similar_fact (threshold applied in SQL), refresh_updated_at, cleanup_stale (14-day retention, FACT_RETENTION_DAYS)
engagement.py       add_signal (atomic decay-and-charge, returns new score), peek_score (read-only decayed score, 0.0 when absent)
thread_history.py   append_turn, get_history (thread-scoped, oldest-first), cleanup_old (60-day retention)
sticker_descriptions.py get_description / save_description — permanent vision-description cache keyed by sticker file_unique_id
embedder.py         embed(text) — fastembed MiniLM-L12 ONNX, returns list[float] (384-dim)
llm_system_prompts.py hash_prompt (pure sha256), ensure (upsert, ON CONFLICT DO NOTHING), cleanup_unreferenced
llm_log.py           insert_call (one response-LLM call per bot message), cleanup_old (60-day retention, RETENTION_DAYS)
message_feedback.py  register, add_reaction / remove_reaction, add_reply, add_correction (no window guard), close_expired_windows, cleanup_old (60-day retention, RETENTION_DAYS)
feedback_metrics.py  recompute_recent_buckets (hourly rollup, idempotent ON CONFLICT DO UPDATE), cleanup_old (365-day retention, RETENTION_DAYS)
```
