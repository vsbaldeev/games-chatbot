# Games Chatbot

A Telegram group chat bot for a Russian-speaking PS5 friend group. Built with a LangGraph pipeline, a LangChain ReAct agent, a custom MCP tool server, and SQLite — deployed on a personal VPS within tight resource constraints.

The bot has a friendly, sarcastic personality — jokes around, roasts chat members, and answers game questions with dry humour. It tracks per-user statistics, maintains conversation memory across sessions, and autonomously reacts to game-related messages.

---

## Message route

Every incoming message (text, voice, photo, video note, sticker, video) passes through
two layers: **stat tracking** first, then the **LangGraph pipeline**.

```
Telegram Update
      │
      ▼
src/bot/handlers.py
  EventHandlerManager   — member tracking, reactions
  CommandHandlerManager — /command handlers
  MessageHandlerManager — text, voice, photo, sticker, video
      │
      ├── stat tracking (achievements module)
      │     • text:  emoji, link, night, forwarded, long_message_max
      │     • voice: voice_messages, voice_max_duration
      │     • photo: photo_messages
      │     • sticker: sticker_messages
      │     • video: video_messages
      │
      └── LangGraph pipeline (src/pipeline/)
            │
            ▼
          router_node (MessageRouter)
            • writes message to unified_messages (text or placeholder)
            • decides should_respond:
                – text:             @mention or reply-to-bot → True
                – voice/video_note/video: 25% random chance
                – photo:            @mention in caption → True, else 25% chance
                                    (real-photo pre-filter: vision LLM call)
            │
            ├─ should_respond=False ──► END  (no reply sent)
            │
            └─ should_respond=True
                  │
                  ▼
                ingester_node (MessageIngester)
                  • text:            processed_text = raw_text
                  • voice:           Groq Whisper transcription
                  • video_note/video: Whisper + PyAV frame extraction
                                       <15s → 1 frame; 15s–2min → 3 frames; >2min → audio only
                                       each frame described by vision LLM
                                       combined as [Аудио]: … / [Видео N/M]: …
                  • photo:           vision LLM one-sentence description
                  all non-text results update unified_messages
                  │
                  ▼
                guard_node (GuardNode)
                  • classifies processed_text with llama-prompt-guard-2-86m
                  • MALICIOUS + explicit trigger (@mention/reply):
                      random refusal (1 of 10) sent, hack attempt recorded
                      in user_memories, short-circuits to END
                  • MALICIOUS + random trigger (25% chance):
                      silently blocked, no message sent, no hack recorded
                  • BENIGN    → continue; fails open on API error
                  │
                  ├─ blocked=True ──► END  (refusal already sent)
                  │
                  └─ blocked=False
                        │
                        ▼
                      context_builder_node (ContextBuilder)
                        • walks reply_to_msg_id chain (max 10 hops) from unified_messages
                        • loads user_memories for every user in the chain
                        • falls back to last 20 messages for flat context fill
                        │
                        ▼
                      agent_node (AgentNode)
                        • builds enriched prompt:
                            [SYSTEM_PROMPT]
                            What I know about people:
                              @user: fact1, fact2 …
                            Reply thread:
                              @user [photo]: <description>
                              @other: lol
                            Recent history: …
                        • invokes Agent.run() → ReAct executor → MCP tools
                        • handles DailyLimitError / RateLimitError
                        • sends reply
                        │
                        ▼
                      memory_writer_node (MemoryWriter)
                        • fires asyncio.create_task() — does NOT block the reply
                        • calls llama-3.1-8b-instant to extract new facts from exchange
                        • upserts up to 3 new facts into user_memories (cap: 10 per user)
```

---

## Agent architecture

```
Agent (src/agent.py)
  │
  ├── MCP subprocess (src/mcp_server.py — stdio transport)
  │     tools/igdb_tools.py    — search_games, get_game_details,
  │                               find_coop_games, find_new_ps5_online_games,
  │                               find_singleplayer_ps_games
  │     tools/store_tools.py   — get_ps_store_sales, get_ps_store_price_tr,
  │                               get_steam_player_count,
  │                               get_steam_app_details, get_steam_reviews_summary
  │     tools/web_tools.py     — web_search (Tavily → DuckDuckGo fallback),
  │                               fetch_article (trafilatura)
  │     tools/media_tools.py   — search_movie_or_tv (TMDB),
  │                               search_anime (AniList),
  │                               get_game_reviews (OpenCritic),
  │                               explain_term (Wikipedia)
  │
  ├── Model fallback chain (AGENT_MODEL_FALLBACKS)
  │     index 0: meta-llama/llama-4-scout-17b-16e-instruct  (500K TPD, primary)
  │     index 1: qwen/qwen3-32b                              (500K TPD, fallback-1)
  │     index 2: openai/gpt-oss-20b                          (200K TPD, fallback-2)
  │     DailyLimitError → advance_model() → rebuild executor with next model
  │     RateLimitError  → exponential back-off (5s, 10s, 20s)
  │     MCP crash (BrokenPipeError/EOFError) → reinit(reset_model=False), retry once
  │
  └── run(chat_id, username, message) — full ReAct agent with tool use
        • loads SQLChatMessageHistory (per chat_id)
        • trims to MAX_HISTORY_MESSAGES before inference
        • trims DB to 40 user messages after save
```

Model index resets to 0 at **00:05 UTC** daily via `ResetModelJobManager`.

---

## Memory architecture

Two complementary memory stores operate independently:

```
┌─────────────────────────────────────────────────────────────┐
│  SQLChatMessageHistory  (LangChain / SQLAlchemy)            │
│  Table: message_store                                        │
│  Key:   chat_id (one session per group chat)                 │
│  Stores: raw HumanMessage + AIMessage turns                  │
│  Cap:   40 user messages per chat (trim_db_history)          │
│  Usage: injected into every agent call as past_messages      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  user_memories  (aiosqlite, src/store/user_memories.py)      │
│  Key:   (chat_id, user_id)                                   │
│  Stores: plain-language facts extracted by MemoryWriter LLM  │
│  Cap:   10 facts per user (oldest pruned on insert)          │
│  Usage: injected into agent prompt by ContextBuilder         │
│         "What I know about people in this chat"              │
│  Writer: llama-3.1-8b-instant, async background task         │
│          (never adds latency to bot reply)                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  unified_messages  (aiosqlite, src/store/unified_messages.py)│
│  Key:   (chat_id, message_id)                                │
│  Stores: every message with content, media_type,             │
│          reply_to_msg_id, file_id                            │
│  Usage: reply-chain walking (ContextBuilder, max 10 hops)    │
│         flat recent-history fill (last 20 messages)          │
│  Written by: router_node (placeholder) + ingester_node       │
│              (real content after transcription/vision)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Database schema

All tables live in a single SQLite file (`data/chat_history.db`).
WAL mode, `synchronous=NORMAL`, `busy_timeout=5000ms`, foreign keys enabled.

```sql
-- LangChain conversation history (managed by SQLAlchemy)
-- one session per chat_id, capped at 40 user messages
message_store (session_id TEXT, message TEXT, ...);

-- Achievement system
CREATE TABLE chat_members (
    chat_id  INTEGER NOT NULL,
    user_id  INTEGER NOT NULL,
    username TEXT,
    PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE user_stats (
    user_id              INTEGER NOT NULL,
    chat_id              INTEGER NOT NULL,
    username             TEXT,
    -- reaction counters
    laugh_reactions      INTEGER NOT NULL DEFAULT 0,
    heart_reactions      INTEGER NOT NULL DEFAULT 0,
    fire_reactions       INTEGER NOT NULL DEFAULT 0,
    thumbsup_reactions   INTEGER NOT NULL DEFAULT 0,
    -- message type counters
    emoji_messages       INTEGER NOT NULL DEFAULT 0,
    sticker_messages     INTEGER NOT NULL DEFAULT 0,
    forwarded_messages   INTEGER NOT NULL DEFAULT 0,
    link_messages        INTEGER NOT NULL DEFAULT 0,
    voice_messages       INTEGER NOT NULL DEFAULT 0,
    video_messages       INTEGER NOT NULL DEFAULT 0,
    video_note_messages  INTEGER NOT NULL DEFAULT 0,
    photo_messages       INTEGER NOT NULL DEFAULT 0,
    night_messages       INTEGER NOT NULL DEFAULT 0,
    -- game stats
    roasted_count        INTEGER NOT NULL DEFAULT 0,
    roulette_win_count   INTEGER NOT NULL DEFAULT 0,
    duel_wins            INTEGER NOT NULL DEFAULT 0,
    long_messages        INTEGER NOT NULL DEFAULT 0,
    -- max-value trackers (use UPDATE … SET col=MAX(col,?) pattern)
    voice_max_duration   INTEGER NOT NULL DEFAULT 0,
    long_message_max     INTEGER NOT NULL DEFAULT 0,
    last_seen            INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, chat_id)
);

CREATE TABLE announced_achievements (
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    key     TEXT    NOT NULL,
    PRIMARY KEY (user_id, chat_id, key)
);

CREATE TABLE message_authors (
    chat_id    INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    username   TEXT    NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);

CREATE TABLE message_reaction_counts (
    chat_id     INTEGER NOT NULL,
    message_id  INTEGER NOT NULL,
    emoji       TEXT    NOT NULL,
    total_count INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (chat_id, message_id, emoji)
);

-- Pipeline stores
CREATE TABLE unified_messages (
    message_id      INTEGER NOT NULL,
    chat_id         INTEGER NOT NULL,
    user_id         INTEGER NOT NULL,
    username        TEXT    NOT NULL,
    content         TEXT    NOT NULL,   -- text / transcript / "[photo] desc"
    media_type      TEXT    NOT NULL DEFAULT 'text',
    reply_to_msg_id INTEGER,
    file_id         TEXT,
    created_at      REAL    NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX idx_unified_messages_chat_time
    ON unified_messages (chat_id, created_at DESC);

CREATE TABLE user_memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    username   TEXT    NOT NULL,
    fact       TEXT    NOT NULL,
    updated_at REAL    NOT NULL
);
CREATE INDEX idx_user_memories_lookup
    ON user_memories (chat_id, user_id);
```

---

## Features

**Game research**
- Ask the bot anything about a game by @mentioning it — it uses IGDB, Steam, OpenCritic, and web search
- Ask for a co-op/online PS5 game recommendation — returns genre, crossplay status, TRY price, PS Store link
- Ask for a singleplayer PS5 game recommendation (IGDB rating ≥ 75)
- Ask about a movie or TV show — TMDB lookup with year, overview, rating, genres
- Ask about an anime — AniList lookup with episode count, score, studios
- Paste an article URL — extracts and summarises the text

**Group utilities**
- Reply by @mentioning the bot or replying to any of its messages
- Bot remembers facts about each user across sessions (per-chat user memories)

**Voice & video**
- Voice messages: 25% chance to transcribe (Groq Whisper) and comment
- Circle video notes and regular videos: 25% chance to transcribe + extract frames (PyAV) for visual understanding
  - <15s → 1 frame; 15s–2min → 3 uniformly distributed frames; >2min → audio only
- Forwarded media is tracked for stats but never responded to

**Photos**
- 25% chance to respond to real photographs (pre-filtered: memes/screenshots skipped)
- @mentioning the bot in a photo caption always triggers a response

**Personality & memory**
- Friendly sarcastic tone in Russian — dry humour, light roasting, no jargon
- Per-chat conversation history (capped at 40 user messages; trimmed to 10 turns for LLM context)
- Per-user long-term facts extracted from conversations and injected into future prompts
- Reply-chain awareness: the bot reads the full thread before responding
- Refuses to discuss sex, drugs, politics, religion, medicine, terrorism, or weapons — redirects in-style
- Prompt injection and jailbreak attempts blocked by Guard Node (llama-prompt-guard-2-86m) before reaching the LLM; repeat offenders tracked in user_memories for roast material

**Roasts**
- `/roast` — on-demand roast of a randomly chosen chat member
- Auto-roast on repeated insults: two consecutive offensive replies to the bot trigger a roast
- Style: short (≤ 2 sentences), sarcastic stand-up comedian; 10% chance of a warm message instead
- Based on the member's recent chat messages

**Russian roulette**
- Daily at 18:00 UTC the bot picks one random chat member: 50% hit, 50% miss
- `/roulette` — on-demand; requires at least 2 registered members
- Survivals tracked and give achievements (up to 50 survivals)

**Gamification**
- Achievements across 18+ tracked stats: reactions, voice/video/photo messages, stickers, links,
  night activity, roasts, roulette survivals, duel wins, long messages; plus silence achievements
  (3, 7, 14, 30 days without a message)
- `/achievements` — last 3 earned achievements with total count
- `/top` — top-3 chat leaderboard by achievement count with medal emojis

**Emoji duel**
- `/duel` — pick an opponent from chat members; first to tap 🔫 wins; 5-minute timeout

---

## Commands

| Command | Description |
|---|---|
| `/dnd_pvp` | D&D adventure, 1 round, PvP |
| `/dnd_coop` | D&D co-op, 2 rounds vs. a boss NPC |
| `/dnd_heist` | The Great Heist — 3 phases: infiltration → job → escape |
| `/duel` | Emoji duel — pick your opponent from chat members |
| `/roulette` | On-demand Russian roulette |
| `/roast` | On-demand прожарка of a randomly chosen chat member |
| `/achievements` | Last 3 earned achievements with total count |
| `/top` | Top-3 chat leaderboard by achievement count |
| `/help` | Command list |
| `/start` | Welcome message |

---

## Project structure

```
src/
├── config.py               env var loading, fails fast on missing required vars
├── agent.py                ReAct agent, MCP client, model fallback chain
├── memory.py               SQLChatMessageHistory wrapper, history trim helpers
├── helpers.py              shared utilities (get_username, is_night_message, …)
├── log.py                  structured logging setup
├── mcp_server.py           MCP stdio server entry point
│
├── bot/                    Telegram bot wiring
│   ├── __init__.py         startup lifecycle, main()
│   ├── __main__.py         python -m src.bot entry point
│   ├── handlers.py         HandlerManagerInterface + Event/Command/MessageHandlerManager
│   └── jobs.py             JobManagerInterface + Roulette/SilenceSweep/ResetModelJobManager
│
├── pipeline/               LangGraph message processing pipeline
│   ├── state.py            BotState, IncomingMessage, AssembledContext TypedDicts
│   ├── router.py           MessageRouter — store + should_respond decision
│   ├── ingester.py         MessageIngester — Whisper transcription, PyAV frame extraction, vision description
│   ├── guard_node.py       GuardNode — prompt injection classifier, hack attempt tracking
│   ├── context_builder.py  ContextBuilder — reply chain, user facts, recent history
│   ├── agent_node.py       AgentNode — enriched prompt assembly, agent invocation
│   ├── memory_writer.py    MemoryWriter — background fact extraction and upsert
│   └── graph.py            LangGraph StateGraph wiring
│
├── store/                  aiosqlite data access layer
│   ├── db.py               single shared connection (WAL, busy_timeout, row_factory)
│   ├── unified_messages.py message store: insert, update_content, get_chain, get_recent
│   └── user_memories.py    fact store: upsert_facts, upsert_hack_attempt, get_facts, get_facts_for_users
│
├── tools/                  MCP tool implementations
│   ├── igdb_tools.py       IGDB: search, details, coop, singleplayer, new releases
│   ├── store_tools.py      PS Store sales + prices, Steam player count/details/reviews
│   ├── web_tools.py        web_search (Tavily/DDG), fetch_article (trafilatura)
│   └── media_tools.py      TMDB movies/TV, AniList anime, OpenCritic reviews, Wikipedia
│
├── events/                 Telegram event handler implementations
│   ├── members.py          track_member, handle_new_chat_members, handle_bot_added_to_chat
│   ├── messages.py         handle_message, handle_voice_message, handle_photo_message,
│   │                       handle_sticker_message, handle_video_message
│   └── reactions.py        handle_reaction
│
├── commands/               command handler implementations
│   ├── general.py          cmd_start, cmd_help
│   ├── statistics.py       cmd_achievements, cmd_top
│   ├── fun/
│   │   ├── roast.py        Roaster class, cmd_roast, generate_roast_text
│   │   └── roulette.py     Roulette class, cmd_roulette, russian_roulette (job)
│   └── games/
│       ├── duel.py         DuelManager class, cmd_duel, handle_duel_callback
│       └── dnd/
│           ├── state.py    LobbyState, ActiveGame, constants
│           ├── llm.py      ScenarioGenerator class
│           ├── views.py    message text builders, edit_safe
│           └── manager.py  DndManager class, cmd_dnd_*, handle_dnd_callback
│
├── achievements/           achievement system
│   ├── definitions.py      Achievement dataclass, ALL_ACHIEVEMENTS, rule tables
│   ├── store.py            DB ops: init_tables, register_member, increment_stat, …
│   └── checker.py          business logic: check_new, check_silence, summaries
│
└── jobs/                   scheduled job implementations
    ├── achievements.py     silence_sweep_job
    └── agent.py            reset_model_job
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Async ecosystem, aiosqlite, type hints |
| Telegram | python-telegram-bot v22 | JobQueue, TypeHandler, native async |
| LLM (agent) | Groq `meta-llama/llama-4-scout-17b-16e-instruct` | 500K TPD; falls back to `qwen/qwen3-32b` then `openai/gpt-oss-20b` |
| LLM (memory) | Groq `llama-3.1-8b-instant` | Fast, cheap; fact extraction in background |
| LLM (roasts) | Groq `llama-3.3-70b-versatile` | Better creative output for прожарки |
| STT | Groq `whisper-large-v3` | 2K RPD free; transcribes voice/video-note messages |
| Vision | Groq `meta-llama/llama-4-scout-17b-16e-instruct` | Photo description and reality-check |
| Security | Groq `meta-llama/llama-prompt-guard-2-86m` | Prompt injection / jailbreak classifier; 14.4K RPD free |
| Video frames | PyAV (`av`) | In-process frame extraction from video/video_note, no subprocess overhead |
| Pipeline | LangGraph StateGraph | Router → Ingester → Guard → ContextBuilder → Agent → MemoryWriter |
| Tool protocol | MCP (stdio) via langchain-mcp-adapters | Clean isolation; tool server restarts independently |
| Game data | IGDB API (Twitch OAuth) | Structured multiplayer/platform/release metadata |
| Store data | Steam public API, psdeals.net RSS | No auth required |
| Media data | TMDB, AniList GraphQL, OpenCritic | Movie/TV/anime/review lookups |
| Web search | Tavily (with key) / DuckDuckGo (fallback) | Recent information beyond training cutoff |
| Storage | SQLite + aiosqlite | Zero-dep, WAL mode, single shared connection |
| Hosting | VPS (Docker) | 4 vCPU / 2 GB RAM, personal server |

---

## Key engineering decisions

**LangGraph pipeline instead of flat handlers.**
Every message flows through a typed `BotState` graph: Router → Ingester → Guard → ContextBuilder → Agent → MemoryWriter. Each node has a single responsibility and can be tested independently. Two short-circuits exist: `should_respond=False` exits after the router for messages the bot ignores; `blocked=True` exits after the guard for injection/jailbreak attempts — neither reaches the main LLM.

**Guard Node with fail-open design.**
`llama-prompt-guard-2-86m` classifies every processed message before the ReAct agent runs. Behavior on `MALICIOUS` depends on how the bot decided to respond (`response_trigger` in `BotState`): if the user explicitly addressed the bot (@mention or reply), a random refusal from a pool of 10 is sent and the hack attempt is recorded in `user_memories` as an incrementing counter fact (`Пытался взломать бота N раз`) — available to `/roast` as material. If the bot picked the message by random chance (25% media trigger), it silently blocks without sending anything — the user wasn't talking to the bot, so a refusal would be confusing. The guard fails open (passes through) if the Groq API is unavailable, so a transient outage never silences the bot.

**Three-model fallback chain.**
The main agent tries `llama-4-scout` first. On `DailyLimitError`, `advance_model()` rebuilds the executor with `qwen/qwen3-32b`, then `openai/gpt-oss-20b` as last resort — all transparently within the same request. MCP crash recovery preserves the current model index rather than resetting to the primary.

**Reply-chain context.**
The bot walks `reply_to_msg_id` links in `unified_messages` up to 10 hops and injects the full thread into the agent prompt. This means replying inside a thread gives the bot full conversational context without relying on the flat message history window.

**Non-blocking memory writes.**
`MemoryWriter` fires `asyncio.create_task()` and returns immediately. The LLM call for fact extraction and the DB upsert happen in the background after the reply is already sent. The user never waits for memory.

**Real-photo pre-filter.**
Before running the full pipeline for a photo message, the handler calls the vision model with a YES/NO "is this a real photograph?" prompt (max 5 tokens). Memes, screenshots, and game art are silently skipped. This avoids wasting the 25% random-chance trigger on non-human photos.

**Single SQLite connection.**
`src/store/db.py` holds one persistent `aiosqlite.Connection` opened at startup. All store modules call `await database.get()` instead of opening new connections. WAL mode lets readers run concurrently with the writer; `aiosqlite.Row` as `row_factory` makes column access by name available everywhere.

**MCP subprocess crash recovery.**
If the tool server dies (OOM, network crash), `Agent.run()` catches `BrokenPipeError`/`EOFError`, calls `init(reset_model=False)` to respawn the subprocess without resetting the model fallback index, and retries once.

---

## Deployment guide

Four external accounts are required: Telegram, Groq, Twitch (for IGDB), and a VPS. All are free.

### 1. Create the Telegram bot

1. Open [@BotFather](https://t.me/BotFather) and send `/newbot`.
2. Choose a display name and a username ending in `bot`.
3. Save the bot token as `TELEGRAM_TOKEN` and the username (with `@`) as `BOT_USERNAME`.

**Disable Privacy Mode** — bots in groups only receive `/` commands by default:

- BotFather → `/mybots` → your bot → **Bot Settings** → **Group Privacy** → **Turn off**

**Make the bot an admin** in your group so it can receive reaction events.

**Register commands** (send `/setcommands` to BotFather, select your bot):

```
dnd_pvp - D&D приключение, 1 раунд, все против всех
dnd_coop - D&D кооп, 2 раунда против босса
dnd_heist - Великое Ограбление — 3 фазы
duel - эмодзи-дуэль между двумя участниками
roulette - русская рулетка
roast - случайный участник получает по заслугам
achievements - последние достижения
top - топ чата по достижениям
help - помощь
```

---

### 2. Get a Groq API key

Sign up at [console.groq.com](https://console.groq.com) (free, no credit card) → **API Keys** → **Create API Key**. Save as `GROQ_API_KEY`.

---

### 3. Get Twitch credentials for IGDB

1. Go to [dev.twitch.tv/console](https://dev.twitch.tv/console) → **Register Your Application**.
2. Set OAuth Redirect URL to `http://localhost`, Category to Application Integration.
3. **Manage** → **New Secret**. Save `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET`.

---

### 4. Optional API keys

| Variable | Service | Default |
|---|---|---|
| `TMDB_API_KEY` | TMDB — movie/TV lookups | `""` (disabled) |
| `TAVILY_API_KEY` | Tavily — web search | `""` (falls back to DuckDuckGo) |

---

### 5. Install Docker and deploy

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

git clone <repo-url> && cd games-chatbot
cp .env.example .env
nano .env   # fill in required vars
chmod 600 .env

docker compose up -d --build
docker compose logs -f
```

Look for `Bot started, all tables and jobs initialized`.

---

### 6. Updates

```bash
git pull && docker compose up -d --build
```

---

### Useful commands

```bash
docker compose logs -f          # stream live logs
docker compose ps               # check container status
docker compose restart          # restart without rebuild
docker compose stop             # stop (keeps data volume)
docker compose down             # remove container, keep volume
docker compose down -v          # remove everything including database
```

### Resource usage

The compose file sets `mem_limit: 800m`. Estimated steady-state RSS is 300–450 MB
(Python process + MCP subprocess), leaving ~1.2 GB free on a 2 GB VPS.
