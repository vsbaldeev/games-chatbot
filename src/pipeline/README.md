LangGraph StateGraph pipeline — processes every incoming Telegram message through typed
nodes. Each node reads BotState and returns a partial update dict.

```
START → router → ingester → filter → guard → context_builder → intent_classifier
                                                                    ├── worker_games ──┐
                                                                    ├── worker_media ──┤
                                                                    └── worker_general ┘
                                                                                       └── response → memory_writer → END
```

Short-circuits:
- `should_respond=False` after router, ingester, or filter → END (no reply)
- `blocked=True` after guard → END (refusal already sent or silently dropped)

## Nodes

```
router          MessageRouter       store message in unified_messages; set should_respond
ingester        MessageIngester     transcribe voice/video (Whisper); describe photos (vision LLM)
filter          MeaninglessFilter   LLM classifier — drop short noise ("ахаха", "lol"); fails open
guard           GuardNode           prompt-injection classifier (llama-prompt-guard-2-86m); fails open
context_builder ContextBuilder      walk reply chain (max 10 hops); load user_memories; last-20 fallback
intent_classifier IntentClassifier  classify as games | media | general; route to matching worker
worker_games    WorkerNode          ReAct agent with IGDB, Steam, PS Store tools
worker_media    WorkerNode          ReAct agent with TMDB, AniList, OpenCritic tools
worker_general  WorkerNode          ReAct agent with web search and article fetch tools
response        ResponseNode        assemble enriched prompt; call main LLM; send reply
memory_writer   MemoryWriter        asyncio.create_task() — extract facts, upsert user_memories
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

## Media ingestion

```
voice                → Whisper transcription → processed_text
video_note / video   → Whisper + PyAV frame extraction
                         duration < 15s   → 1 keyframe (middle)
                         15s – 120s       → 3 keyframes (25%, 50%, 75%)
                         > 120s           → audio only
                       → [Аудио]: <transcript> \n [Видео N/M]: <frame description>
photo                → vision LLM one-sentence description (lazy: only when in reply chain)
                         stored as [photo] placeholder + file_id; described on demand by context_builder
```
