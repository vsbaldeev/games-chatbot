aiosqlite data access layer — single persistent connection shared across all store modules.

All modules call `await database.get()` (src/store/db.py) instead of opening new connections.
WAL mode, `synchronous=NORMAL`, `busy_timeout=5000ms`, `row_factory=aiosqlite.Row`.

## Tables

```sql
-- Per-message store; source of truth for reply-chain resolution
unified_messages (
    message_id      INTEGER,
    chat_id         INTEGER,
    user_id         INTEGER,
    username        TEXT,
    content         TEXT,       -- text / transcript / vision description; [photo] placeholder before enrichment
    media_type      TEXT,       -- "text" | "voice" | "video_note" | "video" | "photo"
    reply_to_msg_id INTEGER,
    file_id         TEXT,       -- Telegram file_id; permanent; used for lazy photo description
    created_at      REAL,
    PRIMARY KEY (chat_id, message_id)
)
INDEX idx_unified_messages_chat_time ON (chat_id, created_at DESC)

-- LLM-extracted facts per user per chat
user_memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER,
    user_id    INTEGER,
    username   TEXT,
    fact       TEXT,
    updated_at REAL
)
INDEX idx_user_memories_lookup ON (chat_id, user_id)

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
    user_id INTEGER,
    chat_id INTEGER,
    key     TEXT,
    PRIMARY KEY (user_id, chat_id, key)
)

-- Tracks registered chat members for job targeting and achievement queries
chat_members (chat_id INTEGER, user_id INTEGER, username TEXT, PRIMARY KEY (chat_id, user_id))

-- LangChain conversation history (managed by SQLAlchemy / SQLChatMessageHistory)
message_store (session_id TEXT, message TEXT, ...)
```

## Modules

```
db.py               single aiosqlite.Connection opened at startup; row_factory=aiosqlite.Row
unified_messages.py insert, update_content, get_chain (max 10 hops), get_recent (last N)
user_memories.py    upsert_facts (cap 10 per user), upsert_hack_attempt, get_facts, get_facts_for_users
```
