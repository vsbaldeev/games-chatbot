LangGraph StateGraph pipeline — processes every incoming Telegram message through typed
nodes. Each node reads BotState and returns a partial update dict.

---

## Overview

```
router ──► ingester ──► filter ──► guard ──► context_builder ──► worker ──► response ──► memory_writer ──► END
```

Conditional exits exist at every node — see the detailed diagrams below.

---

## Router

Stores every message and decides whether the bot should respond.

```
incoming message
    │
    ├─ always: insert into unified_messages
    │              text   → content = raw_text
    │              photo  → content = "[photo]" or "[photo]\n<caption>" (placeholder marks
    │                       row as still needing vision description; caption preserved)
    │              voice / video_note / video → content = placeholder, file_id stored
    │
    ├─ text
    │     ├─ @bot_username in text    → should_respond=True,  trigger="explicit"
    │     ├─ reply to bot message     → should_respond=True,  trigger="explicit"
    │     └─ otherwise               → should_respond=False, trigger="random"
    │
    ├─ voice / video_note / video / photo
    │     ├─ @bot_username in caption → should_respond=True,  trigger="explicit"
    │     ├─ reply to bot message     → should_respond=True,  trigger="explicit"
    │     └─ otherwise               → random.random() < 0.25
    │
    └─ sticker / animation / audio   → should_respond=False (stored as placeholder)
    │
    ├─ should_respond=False + non-forwarded text (≥ 20 chars)
    │       → asyncio.create_task(extract_and_save)   [passive memory, background]
    │         forwarded messages are skipped — channel content must not be
    │         attributed as facts about the person who forwarded it
    │
    ├─ should_respond=False → END
    └─ should_respond=True  → ingester
```

---

## Media processing

Two places process media into text. The ingester handles the current message;
the context builder lazily processes media found in reply chains.

```
ingester (current message, should_respond=True only)
    ├─ text       → processed_text = raw_text
    ├─ voice      → Groq Whisper → transcript
    ├─ video_note → Groq Whisper + frame extraction (see below)
    ├─ video      → Groq Whisper + frame extraction (see below)
    └─ photo      → vision LLM description; combined with caption when present
                    "<description>\n(подпись: <caption>)" form
                    all non-text results: update unified_messages content

frame extraction (PyAV)
    duration < 15s   → 1 keyframe at 50%
    15s – 120s       → 3 keyframes at 25%, 50%, 75%
    > 120s           → audio only, no frames
    output: "[Аудио]: <transcript>\n[Видео 1/N]: <desc>\n[Видео 2/N]: <desc>…"

context_builder lazy enrichment (reply chain, on demand)
    for each photo row in the reply chain that still holds a placeholder:
    [photo] / [photo]\n<caption> → describe_photo(file_id) → combined with caption
                                   → update unified_messages (cached for future replies)
    Detection: content.startswith("[photo]"); after enrichment content begins with the
    description so re-enrichment is skipped automatically.
    Rows without file_id (e.g. old records) are left as-is.
```

---

## Safety (filter → guard)

```
filter  (runs after ingester)
    ├─ media message, processed_text empty
    │       → should_respond=False + asyncio.create_task(react with random emoji)
    ├─ media message, processed_text non-empty → pass through
    ├─ text, raw_text empty  → should_respond=False (silent)
    ├─ text, LLM → MEANINGLESS
    │       → should_respond=False + asyncio.create_task(react with random emoji)
    ├─ text, LLM → MEANINGFUL → should_respond=True
    └─ text, LLM error       → should_respond=True (fails open)
    │
    ├─ should_respond=False → END
    └─ should_respond=True  → guard

guard   llama-prompt-guard-2-86m
    ├─ text empty / BENIGN   → blocked=False → context_builder
    ├─ MALICIOUS + trigger="explicit"
    │       → blocked=True, response = random refusal (pool of 10)
    │       → asyncio.create_task(record_hack_attempt → user_memories)  → END
    ├─ MALICIOUS + trigger="random"
    │       → blocked=True, response=None (silent drop)  → END
    └─ API error             → blocked=False (fails open) → context_builder
```

---

## Response pipeline (context_builder → worker → response → memory_writer)

```
context_builder
    ├─ get_recent(limit=20), excluding the current message — always loaded
    ├─ find replied_to message (from recent window or get_by_id fallback)
    ├─ get_chain(reply_to_msg_id) → reply_chain (max 10 hops, oldest-first)
    ├─ load user_memories facts for all user_ids visible in recent history
    └─ load initiating user's facts if not already in recent participants
    │
    ▼
worker   ReAct agent with all 13 tools (IGDB, Steam, PS Store, TMDB, AniList, web)
    ├─ CONTEXT FIRST: if reply chain already contains the answer, no tools called
    ├─ prompt: reply chain (or recent history for explicit triggers) + current question
    │          random triggers receive only the reply chain — no recent history bleed
    ├─ SearchNotificationCallback sends "🔍 Ищу…" before web_search/fetch_article
    ├─ DailyLimitError → advance_model(), retry with next fallback
    ├─ ContextLengthError → worker_output="" (response node still runs)
    └─ any other error   → worker_output="" (response node still runs)
    │
    ▼
response   main personality LLM
    ├─ thread_id = reply-chain root message_id, or chat_id for flat messages
    ├─ prompt: [SYSTEM] + thread_history (last 10 turns, thread-scoped)
    │            + user facts + recent history (last 10) + replied_to + worker findings + current message
    ├─ DailyLimitError → advance_model(), retry
    ├─ RateLimitError  → exponential backoff (5s, 10s, 20s)
    └─ CJK/Hangul/Thai/Arabic detected → retry with language correction prompt
    │
    ▼
memory_writer
    ├─ is_forwarded=True → skip entirely
    ├─ passive (no response): skip if user_message < 20 chars
    └─ asyncio.create_task() — does NOT block the reply
          → llama-4-scout-17b extracts up to 3 new facts
          → dedup via cosine similarity (fastembed MiniLM-L12, threshold 0.85)
            duplicate → refresh updated_at; new → insert with embedding
          → cap: 30 facts per user per chat, oldest pruned on overflow
          → facts written in Russian
          → cross-user extraction for any @mentioned users (if stripped message ≥ 20 chars)
```

---

## BotState

```python
IncomingMessage:
    chat_id, user_id, username
    raw_text: str | None          # original text or caption
    processed_text: str | None    # transcript / vision description, set by ingester
    media_type: "text" | "voice" | "video_note" | "video" | "photo"
    message_id, reply_to_msg_id, file_id
    is_forwarded: bool            # True when message.forward_origin is set
    media_group_id: str | None    # Telegram album group id

AssembledContext:
    user_facts: dict[str, list[str]]     # username → extracted fact strings
    recent_history: list[dict]           # flat window (last 20), newest-first
    replied_to: dict | None              # the specific message being replied to (for annotation)
    reply_chain: list[dict]              # full reply chain from root to replied-to, oldest-first

BotState:
    incoming: IncomingMessage
    should_respond: bool
    response_trigger: "explicit" | "random"
    blocked: bool
    context: AssembledContext | None
    thread_id: str | None         # reply-chain root message_id or chat_id; scopes LLM history
    worker_output: str | None
    search_notification_msg: Any | None   # Telegram Message used as search indicator
    response: str | None
    context_types: ContextTypes          # Telegram context for sending replies
```
