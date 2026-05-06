LangGraph StateGraph pipeline — processes every incoming Telegram message through typed
nodes. Each node reads BotState and returns a partial update dict.

## Full graph with conditional branches

```
Telegram update
    │
    ▼
router
    ├─ text
    │     ├─ @bot_username in text           → should_respond=True,  trigger="explicit"
    │     ├─ reply to bot message            → should_respond=True,  trigger="explicit"
    │     └─ otherwise                       → should_respond=False, trigger="random"
    │
    ├─ voice / video_note / video / photo
    │     ├─ @bot_username in caption        → should_respond=True,  trigger="explicit"
    │     ├─ reply to bot message            → should_respond=True,  trigger="explicit"
    │     └─ otherwise                       → random.random() < 0.25
    │                                              True  → should_respond=True,  trigger="random"
    │                                              False → should_respond=False, trigger="random"
    │
    └─ other media (sticker, animation, …)  → should_respond=False
    │
    ├─ should_respond=False
    │     ├─ text message (raw_text ≥ 20 chars) ────────────────────────────► memory_writer (passive)
    │     └─ other / short text ────────────────────────────────────────────► END
    │
    └─ should_respond=True
          │
          ▼
        ingester
            ├─ text            → processed_text = raw_text (no LLM call)
            ├─ voice           → Groq Whisper transcription
            │                       → updates unified_messages content
            ├─ video_note      → Groq Whisper + PyAV frame extraction
            │   video          →   duration < 15s  → 1 keyframe (50%)
            │                      15s – 120s      → 3 keyframes (25%, 50%, 75%)
            │                      > 120s          → audio only, no frames
            │                      result: "[Аудио]: <transcript>\n[Видео N/M]: <desc>"
            │                       → updates unified_messages content
            └─ photo           → vision LLM one-sentence description (Russian)
                                   → updates unified_messages content
          │
          ▼
        filter                 (text messages only — media skips LLM classification)
            ├─ media_type != "text"
            │     ├─ processed_text is empty → should_respond=False  (logs warning)
            │     │                             asyncio.create_task(react with random emoji)
            │     └─ processed_text non-empty → pass through unchanged
            │
            └─ media_type == "text"
                  ├─ raw_text empty           → should_respond=False  (silent)
                  ├─ LLM says MEANINGLESS     → should_respond=False
                  │                              asyncio.create_task(react with random emoji)
                  ├─ LLM says MEANINGFUL      → should_respond=True
                  └─ LLM error               → should_respond=True   (fails open)
          │
          ├─ should_respond=False ──────────────────────────────────────────────► END
          │
          └─ should_respond=True
                │
                ▼
              guard             llama-prompt-guard-2-86m classifies processed_text
                ├─ text empty                → blocked=False, pass through
                ├─ BENIGN                    → blocked=False, pass through
                ├─ MALICIOUS + trigger="explicit"
                │     → blocked=True
                │     → response = random refusal from pool of 10
                │     → asyncio.create_task(record_hack_attempt → user_memories)
                └─ MALICIOUS + trigger="random"
                │     → blocked=True, response=None (silent drop — user wasn't addressing bot)
                └─ API error                 → blocked=False (fails open)
                │
                ├─ blocked=True  (response node sends refusal if response is set) ──► END
                │
                └─ blocked=False
                      │
                      ▼
                    context_builder
                        ├─ walk reply_to_msg_id chain in unified_messages (max 10 hops)
                        │     └─ for each photo placeholder in chain:
                        │           → describe_photo(file_id) → update unified_messages
                        ├─ load user_memories facts for every user_id in chain
                        └─ if initiating user not in chain facts:
                              → load their facts separately
                        │   recent_history = get_recent(limit=20)  (fallback context fill)
                        │
                        ▼
                      intent_classifier     llama-4-scout, max_tokens=5, temp=0
                          ├─ "games"   → worker_games   (IGDB, Steam, PS Store tools)
                          ├─ "media"   → worker_media   (TMDB, AniList, OpenCritic tools)
                          └─ "general" → worker_general  (web_search, fetch_article tools)
                          │
                          ▼
                        worker_*            ReAct agent, same model fallback chain as response
                            ├─ builds input: reply chain (or recent history) + question
                            ├─ invokes tools until answer is assembled
                            ├─ SearchNotificationCallback sends "🔍 Ищу…" before web_search
                            ├─ DailyLimitError → advance_model(), retry with next fallback
                            └─ any other error → worker_output=""  (response still runs)
                          │
                          ▼
                        response            main personality LLM
                            ├─ assembles prompt:
                            │     [SYSTEM_PROMPT]
                            │     What I know about people: @user: fact1, fact2 …
                            │     Worker findings: <worker_output>
                            │     Reply thread / Recent history
                            │     Current message
                            ├─ SQLChatMessageHistory trimmed to last 10 turns for context
                            ├─ DailyLimitError → advance_model(), retry
                            ├─ RateLimitError  → exponential backoff (5s, 10s, 20s)
                            └─ foreign-script check: CJK/Hangul/Thai/Arabic detected
                                  → retry once with language correction prompt
                          │
                          ▼
                        memory_writer
                            └─ asyncio.create_task() — does NOT block reply
                                  → llama-3.1-8b-instant extracts new facts
                                  → upsert up to 3 facts into user_memories (cap: 10 per user)
                            note: only runs when the bot actually replied — memory extraction
                                  needs both sides of the exchange (user message + bot response)
                                  to derive meaningful facts. messages the bot ignored are still
                                  stored in unified_messages for reply-chain context.
                          │
                          ▼
                         END
```

## BotState

```python
IncomingMessage:
    chat_id, user_id, username
    raw_text: str | None          # original text or caption
    processed_text: str | None    # transcript / vision description, set by ingester
    media_type: "text" | "voice" | "video_note" | "video" | "photo"
    message_id, reply_to_msg_id, file_id

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
