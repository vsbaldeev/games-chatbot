# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Testing policy

Do not write tests unless the user explicitly asks for them.

## Running the bot

```bash
# Local development (requires .env populated from .env.example)
python -m src.bot

# Production
docker compose up -d --build
docker compose logs -f
```

Required env vars: `TELEGRAM_TOKEN`, `GROQ_API_KEY`, `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`, `BOT_USERNAME`.  
Optional: `TAVILY_API_KEY`, `TMDB_API_KEY`.

## Tests

```bash
pytest src/pipeline/test_filter_node.py          # run the filter node test suite
pytest src/pipeline/test_filter_node.py -k name  # run a single test
```

Tests use `pytest-asyncio`. There is currently only one test file.

## Architecture

### Message pipeline (LangGraph)

Every Telegram message is processed by a `StateGraph` compiled once at startup in `src/pipeline/graph.py`. The graph is stored on the `Agent` singleton and retrieved via `agent.get_pipeline()`.

```
START → router → ingester → filter → guard → context_builder → intent_classifier
                                                                   ├── worker_games ──┐
                                                                   ├── worker_media ──┤
                                                                   └── worker_general ┘
                                                                                      └── response → memory_writer → END
```

Early exits via `END` happen when `should_respond=False` (after router, ingester, filter) or `blocked=True` (after guard). Photos always reach the ingester even when `should_respond=False` so their vision description is stored and available if someone later replies to the photo and @mentions the bot.

Each node lives in its own file under `src/pipeline/` and receives/returns a partial `BotState` dict (defined in `state.py`).

### Pipeline nodes

| Node | File | Responsibility |
|---|---|---|
| `router` | `router.py` | Decides `should_respond` (explicit @mention/reply=always, other media=25% random), stores message in `unified_messages` |
| `ingester` | `ingester.py` | Transcribes voice/video via Whisper; extracts frames from videos <2 min; describes photos with vision LLM |
| `filter` | `filter_node.py` | LLM classifier — drops short noise like "ахаха" or "lol" to save tokens; fails open |
| `guard` | `guard_node.py` | Prompt-injection classifier (`llama-prompt-guard-2-86m`); sets `blocked=True` and records hack attempts in `user_memories` for roast material; fails open |
| `context_builder` | `context_builder.py` | Walks `reply_to_msg_id` chain (max 10 hops) from `unified_messages`; loads `user_memories` facts for all participants; falls back to last 20 messages |
| `intent_classifier` | `intent_node.py` | Classifies message as `games`, `media`, or `general`; routes to the matching worker |
| `worker_*` | `worker_node.py` | Three specialist ReAct agents with domain tool sets — games (IGDB, Steam, PS Store), media (TMDB, AniList, OpenCritic), general (web search, article fetch) |
| `response` | `response_node.py` | Main LLM call assembling the final reply from system prompt, user facts, reply thread, worker output |
| `memory_writer` | `memory_writer.py` | `asyncio.create_task()` — non-blocking extraction of new user facts via `llama-3.1-8b-instant`, upserted to `user_memories` (cap: 10 per user per chat) |

### LLM / model fallback

`src/agent.py` holds the `Agent` singleton. The primary model is `meta-llama/llama-4-scout-17b-16e-instruct`. On `DailyLimitError` the model index advances through a fallback chain (`qwen/qwen3-32b` → `openai/gpt-oss-20b`). On `RateLimitError` it retries with exponential backoff (5 s, 10 s, 20 s). The model index resets to 0 daily at 00:05 UTC via `ResetModelJobManager`.

### Scheduled jobs

All jobs are registered in `src/bot/jobs.py` and wired in `src/bot/__init__.py`.

| Job | Schedule | What it does |
|---|---|---|
| `RoastJobManager` | Daily 12:00 UTC | Runs `weekly_roast_job`; the job itself exits early unless today matches this week's deterministic random roast day (seeded by ISO year+week) |
| `SilenceSweepJobManager` | Daily 10:00 UTC | Awards silence achievements to users inactive for 7/14/30 days |
| `ResetModelJobManager` | Daily 00:05 UTC | Resets the LLM fallback index to 0 |

### Achievement system

`src/achievements/definitions.py` is the single source of truth for what can be earned. `ACHIEVEMENT_RULES` maps DB stat columns to `(threshold, achievement_key)` pairs. `SILENCE_THRESHOLDS` is separate (checked by `silence_sweep_job` rather than on every message).

`TRACKABLE_STATS` and `MAX_TRACKABLE_STATS` in `definitions.py` act as allowlists — `store.increment_stat()` and `store.update_max_stat()` raise `ValueError` for anything not in those sets.

New achievements are announced via `achievements.notify_unlocks()` called at the end of every event handler that increments a stat.

### Database

Single SQLite file at `data/chat_history.db` (WAL mode). A single `aiosqlite` connection is opened at startup and reused everywhere via `src/store/db.py`. Schema is initialized and migrated in `src/achievements/store.py:init_tables()`.

Key tables:
- `unified_messages` — every message processed by the pipeline, including vision descriptions; indexed by `(chat_id, created_at DESC)`
- `user_memories` — extracted facts per user per chat; indexed by `(chat_id, user_id)`
- `user_stats` — cumulative counters and max-value stats per user per chat
- `announced_achievements` — deduplication log so each achievement notifies only once
- `message_store` — LangChain conversation history (managed by `SQLChatMessageHistory`)
- `sent_memes` — dedup log of meme URLs already sent per chat; primary key `(chat_id, url)`; initialized by `src/memes/store.py:init_table()`

### Bot wiring

`src/bot/handlers.py` contains three `HandlerManager` classes registered in `src/bot/__init__.py`:
- `EventHandlerManager` — member tracking, reactions
- `CommandHandlerManager` — all `/commands`
- `MessageHandlerManager` — text, voice, video_note, photo, sticker, video, animation

Adding a new command requires: handler function → register in `CommandHandlerManager` → export from `src/commands/__init__.py` → mention in `cmd_help` and the agent system prompt in `src/agent.py`.

### Meme feature (`/meme`)

`src/memes/` — fetches a random image post from Reddit and sends it with its post title as caption.

| File | Responsibility |
|---|---|
| `src/memes/fetcher.py` | Calls Reddit JSON API for each subreddit in `SUBREDDITS`, extracts `(url, title)` pairs, excludes already-seen URLs, picks a random candidate |
| `src/memes/store.py` | `sent_memes` table — tracks which URLs have been sent per chat to prevent repeats |
| `src/commands/fun/meme.py` | `cmd_meme` handler — calls the fetcher, sends `reply_photo` with caption |

Subreddits scraped: `ru_memes`, `expectedrussians`, `ruAsska`, `Pikabu`. All use Reddit's public JSON endpoint (`/hot.json?limit=100`) — no API key required, only a `User-Agent` header. A subreddit failure is logged as a warning and skipped; the rest still proceed. The pool is per-chat and never resets automatically — once all fetched posts have been sent, the command replies with a text message.
