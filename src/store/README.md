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
                                     -- "document" | "location" | "contact" | "dice" | "venue" | "poll"
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

-- One row per bot message of any source — which reaction emoji landed, and
-- whether any feedback arrived at all, within the first 15 minutes after
-- sending (every mutator's WHERE clause enforces that window). Per-reply
-- detail is not here: it lives one row per reply in bot_reply_labels below.
-- v5_0004 dropped the four counters this table used to carry (reactions,
-- replies, corrections, thread_depth) — each was read by exactly one of the
-- eight bot_feedback_metrics columns dropped in the same migration, so after
-- that the bot would have written four numbers nothing could read.
-- See src/feedback/README.md. Retention: 60 days.
bot_message_feedback (
    chat_id           BIGINT      NOT NULL,
    message_id        BIGINT      NOT NULL,
    source            TEXT        NOT NULL,   -- "pipeline" | "meme" | "selfie" | "roles" | "group_profile" | "notice"
    trigger           TEXT        NOT NULL DEFAULT '-',
    filter_verdict    TEXT        NOT NULL DEFAULT '-',
    sent_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    emojis            JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- {"🤡": 1, "😂": 2}; feeds the two emoji rates
    first_feedback_at TIMESTAMPTZ,            -- load-bearing: close_expired_windows derives `ignored` from it
    ignored           BOOLEAN,                -- NULL while the 15-minute window is still open
    PRIMARY KEY (chat_id, message_id)
)
INDEX idx_bot_message_feedback_sent ON (sent_at)

-- One row per reply to a bot message, carrying how that reply was received.
-- Written by track_reply (src/feedback/replies.py) with reaction = NULL, then
-- labelled offline by reply_labels_job (src/jobs/reply_labels.py) every 15
-- minutes. Exactly one row per reply, attributed to its nearest bot ancestor,
-- even when the reply chain passes through several bot messages.
--
-- reaction IS NULL means not yet labelled, and the partial index on that is
-- the job's work queue and its only read. Emoji reactions are NOT rows here:
-- they have no message and are aggregated straight from
-- bot_message_feedback.emojis in the rollup. Retention: 60 days.
bot_reply_labels (
    chat_id          BIGINT      NOT NULL,
    reply_message_id BIGINT      NOT NULL,   -- the member's reply
    bot_message_id   BIGINT      NOT NULL,   -- nearest bot ancestor in the chain
    depth            INTEGER     NOT NULL,   -- hop distance from the reply to that ancestor
    media_type       TEXT        NOT NULL,   -- as unified_messages stores it; non-"text" resolves to MEDIA by rule
    reaction         TEXT,                   -- POSITIVE | NEGATIVE | NEUTRAL | UNCLEAR | MEDIA | SKIPPED; NULL = unlabelled
    cause            TEXT,                   -- PERSONA | INVENTED_LIFE | FACTUAL | TONE | IRRELEVANT | UNCLEAR
                                             -- only on a NEGATIVE text reply; stored and never aggregated
    labelled_by      TEXT,                   -- "model" | "media_rule" | "empty_text" | "excluded_trigger" | "unlabelled" | "human"
    attempts         INTEGER     NOT NULL DEFAULT 0,  -- model attempts spent; capped at MAX_ATTEMPTS = 3
    labelled_at      TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (chat_id, reply_message_id)
)
INDEX idx_bot_reply_labels_pending     ON (created_at) WHERE reaction IS NULL  -- the work queue
INDEX idx_bot_reply_labels_created     ON (created_at)                         -- retention sweep
INDEX idx_bot_reply_labels_bot_message ON (chat_id, bot_message_id) INCLUDE (reaction, media_type)
                                       -- v5_0005: makes the rollup's five EXISTS subqueries index-only

-- Hourly rollup of bot_message_feedback, written by feedback_metrics_job
-- (src/jobs/feedback.py). The only feedback table dashboards should query —
-- counts sit next to their rates so a daily/weekly figure is
-- SUM(count) / SUM(messages), never an average of hourly rates.
--
-- v5_0004 dropped eight columns outright (corrected, correction_rate, reacted,
-- reaction_rate, replied, reply_rate, avg_thread_depth,
-- median_first_feedback_s) and added the fourteen below. The drop destroyed
-- the history in those eight — they do not exist and are not NULL. "Any
-- reaction" and "any reply" stopped being questions anyone reads once the
-- signal was split by sentiment; the last two were off-subject. messages,
-- ignored and ignore_rate stayed, because ignore_rate is one of the seven
-- reported metrics and is computed from feedback.ignored.
--
-- The seven rates share one denominator (messages) so they can be read side by
-- side, and they are deliberately not mutually exclusive: one bot message can
-- draw a 👍 and an angry reply. Read them split by trigger — see
-- src/feedback/README.md — and never without unlabelled_rate beside them.
-- Rows on trigger = 'social_link' are not written at all (excluded from
-- measurement), so that trigger has no buckets rather than zero-filled ones.
-- Retention: 365 days.
bot_feedback_metrics (
    bucket_start             TIMESTAMPTZ NOT NULL,  -- date_trunc('hour', sent_at)
    chat_id                  BIGINT      NOT NULL,
    source                   TEXT        NOT NULL,
    trigger                  TEXT        NOT NULL,
    messages                 INTEGER     NOT NULL,  -- the shared denominator
    ignored                  INTEGER     NOT NULL,
    ignore_rate              REAL        NOT NULL,
    -- counts, nullable with no default: buckets written before v5_0004 have
    -- no honest value for them
    media_responded          INTEGER,               -- >=1 non-text reply
    positive_emoji           INTEGER,               -- >=1 reaction in ❤👍😂🔥💯😅😁
    negative_emoji           INTEGER,               -- >=1 reaction outside that set
    positive_responded       INTEGER,               -- >=1 text reply labelled POSITIVE
    negative_responded       INTEGER,               -- >=1 text reply labelled NEGATIVE
    neutral_responded        INTEGER,               -- >=1 text reply labelled NEUTRAL
    unlabelled               INTEGER,               -- >=1 text reply still NULL or UNCLEAR
    -- rates, each the count above over `messages`
    media_response_rate      REAL,
    positive_emoji_rate      REAL,
    negative_emoji_rate      REAL,
    positive_response_rate   REAL,
    negative_response_rate   REAL,
    neutral_response_rate    REAL,
    unlabelled_rate          REAL,                  -- the error bar on the three response rates
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
message_feedback.py  register, add_reaction / remove_reaction, add_reply (no depth argument — per-reply detail is in reply_labels.py), close_expired_windows, cleanup_old (60-day retention, RETENTION_DAYS). add_correction is gone with correction tracking (v5_0004)
reply_labels.py      enqueue (idempotent ON CONFLICT DO NOTHING), fetch_pending (oldest-first work queue, joins the reply text, the bot message and its source/trigger so the job needs no per-row query; skips rows at MAX_ATTEMPTS), write_label (guarded by `reaction IS NULL`, so a human label is never overwritten), bump_attempts, cleanup_old (60-day retention, RETENTION_DAYS). MAX_ATTEMPTS = 3
feedback_metrics.py  recompute_recent_buckets (hourly rollup, idempotent ON CONFLICT DO UPDATE; the positive emoji set is bound as a parameter from src/feedback/sentiment.py rather than inlined), cleanup_old (365-day retention, RETENTION_DAYS)
```
