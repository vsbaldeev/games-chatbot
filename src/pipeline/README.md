LangGraph StateGraph pipeline — processes every incoming Telegram message through typed
nodes. Each node reads BotState and returns a partial update dict.

---

## Overview

```
router ──► ingester ──► filter ──► guard ──► context_builder ──► intent_classifier
                                                                         │
                                               ┌─────────────────────────┼─────────────────────┐
                                               ▼                         ▼                     ▼
                                          worker_games            worker_media           worker_general
                                               └─────────────────────────┼─────────────────────┘
                                                                         ▼
                                                                      response ──► memory_writer ──► END
```

Conditional exits exist at every node — see the detailed diagrams below.

---

## Router

Stores every message and decides whether the bot should respond.

```
incoming message
    │
    ├─ always: insert into unified_messages
    │              text   → content = raw_text (or caption)
    │              photo  → content = caption if present, else [photo]
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
    └─ other (sticker, animation, …) → should_respond=False
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
    └─ photo      → vision LLM, one-sentence Russian description
                    all non-text results: update unified_messages content

frame extraction (PyAV)
    duration < 15s   → 1 keyframe at 50%
    15s – 120s       → 3 keyframes at 25%, 50%, 75%
    > 120s           → audio only, no frames
    output: "[Аудио]: <transcript>\n[Видео 1/N]: <desc>\n[Видео 2/N]: <desc>…"

context_builder lazy enrichment (reply chain, on demand)
    for each row in the reply chain that still holds a placeholder:
    [photo]      → describe_photo(file_id)              → update unified_messages
    [voice]      → transcribe_voice(file_id)            → update unified_messages
    [video_note] → transcribe_video(file_id)  (+ frames) → update unified_messages
    [video]      → transcribe_video(file_id)  (+ frames) → update unified_messages
    Results are cached — repeated replies reuse the stored description/transcript.
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

## Response pipeline (context_builder → response → memory_writer)

```
context_builder
    ├─ walk reply_to_msg_id links (max 10 hops) → enrich media placeholders (see above)
    ├─ load user_memories facts for all user_ids in chain
    └─ load initiating user's facts if not already in chain
    │  recent_history = get_recent(limit=20)  [fallback when no reply chain]
    │
    ▼
intent_classifier   llama-4-scout, max_tokens=5, temp=0
    ├─ "games"   → worker_games   (IGDB, Steam, PS Store tools)
    ├─ "media"   → worker_media   (TMDB, AniList, OpenCritic tools)
    └─ "general" → worker_general (web_search, fetch_article tools)
    │
    ▼
worker_*   ReAct agent, same model fallback chain as response node
    ├─ prompt: reply chain (or recent history) + current question
    ├─ SearchNotificationCallback sends "🔍 Ищу…" before web_search calls
    ├─ DailyLimitError → advance_model(), retry with next fallback
    └─ any other error → worker_output="" (response node still runs)
    │
    ▼
response   main personality LLM
    ├─ prompt: [SYSTEM] + SQLChatMessageHistory (last 10 turns)
    │            + user facts + worker findings + reply chain + current message
    ├─ DailyLimitError → advance_model(), retry
    ├─ RateLimitError  → exponential backoff (5s, 10s, 20s)
    └─ CJK/Hangul/Thai/Arabic detected → retry with language correction prompt
    │
    ▼
memory_writer
    ├─ is_forwarded=True → skip entirely
    └─ asyncio.create_task() — does NOT block the reply
          → llama-4-scout-17b extracts up to 3 new facts
          → dedup via cosine similarity (fastembed MiniLM-L12, threshold 0.85)
            duplicate → refresh updated_at; new → insert
          → cap: 30 facts per user per chat, oldest pruned on overflow
          → facts written in Russian
          → cross-user extraction for any @mentioned users
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

AssembledContext:
    reply_chain: list[dict]              # messages up the reply chain, oldest first
    user_facts: dict[str, list[str]]     # username → extracted fact strings
    recent_history: list[dict]           # flat window (last 20) for context fill

BotState:
    incoming: IncomingMessage
    should_respond: bool
    response_trigger: "explicit" | "random"
    blocked: bool
    context: AssembledContext | None
    intent: "games" | "media" | "general" | None
    worker_output: str | None
    response: str | None
    context_types: ContextTypes          # Telegram context for sending replies
```
