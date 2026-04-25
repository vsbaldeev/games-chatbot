# Games Chatbot

A Telegram group chat bot for a Russian-speaking PS5 friend group. Built with a LangChain ReAct agent, a custom MCP tool server, and SQLite — deployed on a personal VPS within tight resource constraints.

The bot has a friendly, sarcastic personality — jokes around, roasts chat members, and answers game questions with dry humour. It tracks per-user statistics, maintains conversation memory across sessions, and autonomously reacts to game-related messages.

---

## Architecture

```
Telegram Group
      │
      ▼
python-telegram-bot v22          ← polling, JobQueue, TypeHandler
      │
      ├── Command handlers        ← /coop, /play, /games …
      ├── MessageHandler (text)   ← keyword/mention trigger, 60 s cooldown
      ├── MessageHandler (voice)  ← voice & video-note transcription, 50% chance
      ├── TypeHandler (group=-1)  ← passive member tracking
      └── JobQueue                ← daily roast 09:00 MSK, roulette 15:00 MSK, sale check 10:00 MSK
      │
      ▼
Two-tier LLM routing
  ├── Direct (@mention / reply to bot) → LangChain ReAct Agent (full tool use)
  └── Keyword-triggered (passive)      → Lightweight model (no tools, fast)
      │
      ▼
LangChain ReAct Agent
  model : meta-llama/llama-4-scout-17b-16e-instruct (primary)
          → qwen/qwen3-32b (fallback on daily limit)
          → openai/gpt-oss-20b (final fallback)
  memory: RunnableWithMessageHistory (per chat_id, last 10 turns for LLM context)
  retries: exponential back-off on 429, one-shot reinit on MCP crash
      │
      ▼
MCP stdio subprocess  (src/mcp_server.py)
  ├── search_games(query)                → IGDB Apicalypse API
  ├── get_game_details(game_id)          → IGDB (multiplayer_modes, platforms)
  ├── get_steam_player_count(name)       → Steam Store + ISteamUserStats
  ├── find_coop_games(player_count, offset) → IGDB, PS5 platform ID 167; offset paginates results
  ├── find_new_ps5_online_games(days)    → IGDB, recent PS5 multiplayer releases
  ├── get_ps_store_sales(limit)          → psdeals.net RSS, current sale titles
  └── get_ps_store_price_tr(game_name)   → Turkish PS Store link + TRY price via psdeals.net

SQLite  (aiosqlite, WAL mode, busy_timeout=5 s)
  ├── message_store      LangChain SQLChatMessageHistory (capped: 40 user msgs per chat)
  ├── wishlists          per-user game wishlist (used for sale alerts)
  ├── chat_members       member registry for achievements and roasts
  ├── user_stats         per-user counters (7 tracked dimensions)
  └── announced_sales    dedup table, 7-day TTL
```

---

## Features

**Game research**
- `/games` — rotates across 4 distinct queries each call: new PS5 multiplayer releases, current PS Store discounts, combined new+sale overlap, and most-alive games by Steam player count; every result includes crossplay status, player count, TRY price, and a direct Turkish PS Store link
- `/coop` — finds one PS5 co-op game for 3–8 players using a random IGDB offset so suggestions vary each call; includes crossplay info, TRY price, and Turkish PS Store link
- Search games and fetch details via IGDB (platforms, genres, multiplayer modes, rating)
- Recent PS5 multiplayer releases via IGDB filtered by release date and platform
- Current PS Store sales via psdeals.net RSS, cross-referenced with IGDB multiplayer data
- Live Steam player count via the public ISteamUserStats API
- Turkish PS Store prices fetched from psdeals.net; always returns a `store.playstation.com/tr-tr` search link as fallback

**Group utilities**
- Daily evening nudge at 21:00 MSK — bot sends a random "who's playing tonight?" message to all chats
- Daily PS Store sale scan via [psdeals.net](https://psdeals.net) RSS; notifies chats when wishlist games are discounted (7-day deduplication)
- Reply by @mentioning the bot or replying to any of its messages — handles any question directly

**Voice & video**
- Bot listens to voice messages and circle video notes (25% chance in groups; 100% when bot is @mentioned in the caption)
- Transcribes via Groq `whisper-large-v3-turbo`, replies with a comment (no transcript echo)
- Always responds in private chat

**Personality & memory**
- Friendly sarcastic tone in Russian — dry humour, light roasting, no jargon from specific wikis or subcultures
- Per-chat conversation memory (SQLite, capped at 40 user messages; trimmed to 10 turns for LLM context)
- Keyword substring matching — responds when any game/tech keyword appears anywhere in a message
- Two-tier routing: @mentions and replies use the full agent with tool access; passive keyword triggers use a lightweight model (llama-3.1-8b-instant, 500K TPD) to save daily quota
- Refuses to discuss politics, religion, or drugs — redirects in-style
- Anti-prompt-injection: mocks "forget your instructions" attempts

**Прожарка (roasts)**
- Daily morning прожарка at 09:00 MSK — comedian-style roast or morning forecast for a random member
- `/prozharka` — on-demand roast of a randomly chosen eligible member; 2-minute per-chat cooldown
- Roast style: short (≤ 3 sentences), sarcastic stand-up comedian; 10% chance of a warm supportive message instead
- Content source: based on the member's recent chat messages or a random world theme (50/50)
- Members are mentioned with `@username` in roasts
- Each member can be roasted at most twice per day (shared limit between daily job and `/prozharka`)

**Russian roulette**
- Daily at 15:00 MSK the bot picks one random chat member and fires — 50% chance to hit, 50% chance to miss
- Both outcomes have distinct dramatic announcements (8 hit variants, 7 miss variants)
- Requires at least 2 members in the chat

**Gamification**
- 12 achievements across 7 tracked stats: crossplay queries, tech explanations, night messages, research, co-op searches, session polls, sale notifications
- `/achievements [all]` — individual or full-chat badge board
- Rank system: 8 tiers from "Только распаковал PS5" 📦 to "Батя чата" 👑; points earned from all activity dimensions weighted by value
- `/rank` — personal rank card with point breakdown and progress to next tier
- `/top` — full-chat leaderboard sorted by points with medal emojis for top 3

---

## Commands

| Command | Description |
|---|---|
| `/games` | New PS5 online releases, current sales, crossplay & player count — angle rotates each call |
| `/coop` | Find one PS5 co-op game for 3–8 players (exclusive or crossplay) |
| `/achievements [all]` | Badge board |
| `/rank` | Personal rank card with point breakdown and next-tier progress |
| `/top` | Full-chat leaderboard sorted by points |
| `/prozharka` | On-demand прожарка of a randomly chosen chat member |
| `/help` | Command list |

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Async ecosystem, aiosqlite, type hints |
| Telegram | python-telegram-bot v22 | JobQueue, TypeHandler, native async |
| LLM (agent) | Groq `meta-llama/llama-4-scout-17b-16e-instruct` | 500K TPD, 30K TPM; falls back to `qwen/qwen3-32b` then `openai/gpt-oss-20b` on daily limit |
| LLM (passive) | Groq `llama-3.1-8b-instant` | 500K TPD, 14.4K RPD; keyword-triggered replies without tool use |
| LLM (roasts) | Groq `llama-3.3-70b-versatile` | 100K TPD; better creative output for прожарки |
| STT | Groq `whisper-large-v3-turbo` | 2K RPD free; transcribes voice/video-note messages |
| Agent | LangChain ReAct + LangGraph | Tool-use loop with bounded iterations |
| Tool protocol | MCP (stdio) via langchain-mcp-adapters | Clean isolation; tool server restarts independently |
| Game data | IGDB API (Twitch OAuth) | Structured multiplayer/platform/release metadata |
| Player data | Steam public API | No auth required |
| Sale data | psdeals.net RSS | Free, covers all regions |
| Storage | SQLite + aiosqlite | Zero-dep, WAL mode for concurrent readers |
| Hosting | VPS (systemd) | 4 vCPU / 2 GB RAM, personal server |

---

## Key engineering decisions

**Three-model fallback chain.**
The main agent tries `meta-llama/llama-4-scout-17b-16e-instruct` (500K TPD) first. On `DailyLimitError`, `__advance_model()` rebuilds the executor with `qwen/qwen3-32b`, then `openai/gpt-oss-20b` as a last resort — all transparently within the same request. MCP crash recovery preserves the current model index rather than resetting to the primary.

**Two-tier LLM routing.**
Not every message needs the full ReAct agent. Keyword-triggered passive responses (user wasn't addressing the bot) are handled by `run_lightweight` — a direct `ChatGroq.ainvoke` call without any tool graph. This keeps the expensive daily quota for tool-use tasks and commands. On quota errors the lightweight path silently skips (no error shown, since the user wasn't asking the bot directly).

**Agent built once, not per request.**
`ChatGroq`, the prompt template, `AgentExecutor`, and `RunnableWithMessageHistory` are constructed in `init_agent()` at startup and reused. This avoids repeated allocation of LangChain's graph structures on every message — important on a 2 GB VPS.

**MCP subprocess crash recovery.**
If the tool server dies (OOM, network crash), `run_agent` catches `BrokenPipeError` / `EOFError`, calls `init_agent(reset_model=False)` to respawn the subprocess without resetting the model fallback index, and retries once.

**Two-level history trimming.**
`trim_history` caps the LLM context window at `MAX_HISTORY_MESSAGES` (10) total messages before each inference call. `trim_db_history` caps the SQLite table at 40 user messages per chat after each save — prevents unbounded DB growth while keeping enough context for roast generation and conversation continuity.

**Sync SQLAlchemy wrapped in `asyncio.to_thread`.**
`SQLChatMessageHistory` uses blocking SQLAlchemy. History reads and writes run entirely inside a thread-pool worker so the event loop is never blocked.

**SQLite WAL mode across all tables.**
Two different libraries (`aiosqlite` and SQLAlchemy from LangChain) share one database file. WAL mode lets readers run concurrently with the writer and eliminates the busy-lock contention that default journal mode causes.

---

## Project structure

```
src/
├── config.py          env var loading and validation (fails fast at startup)
├── mcp_server.py      standalone MCP server — IGDB + Steam + psdeals tools, stdio transport
├── agent.py           LangChain ReAct agent, model fallback chain, lightweight path, retry logic
├── memory.py          SQLChatMessageHistory wrapper, LLM-context trim, DB-level trim
├── bot.py             all Telegram handlers, jobs, startup
├── wishlist.py        wishlist CRUD (aiosqlite) — used by sale alert job
├── psstore.py         psdeals.net RSS fetch, sale dedup, announced_sales table
├── achievements.py    12 achievements, 7 tracked stats, migration helper
└── ranks.py           8-tier rank system: point computation, leaderboard, breakdown
```

---

## Deployment guide

Four external accounts are required: Telegram, Groq, Twitch (for IGDB), and a VPS. All are free.

### 1. Create the Telegram bot

1. Open [@BotFather](https://t.me/BotFather) in Telegram and send `/newbot`.
2. Choose a display name (e.g. `Games Bot`) and a username ending in `bot` (e.g. `@MyGamesBot`).
3. BotFather replies with your **bot token** — save it as `TELEGRAM_TOKEN`.
4. Save the full username (with `@`) as `BOT_USERNAME`.

**Disable Privacy Mode** — by default bots in groups only receive messages that start with `/`. To let the bot read all messages:

- In BotFather: `/mybots` → select your bot → **Bot Settings** → **Group Privacy** → **Turn off**

**Register commands** so Telegram shows autocomplete in the chat. Send `/setcommands` to BotFather, select your bot, then paste:

```
games - свежие игры для PS5: новинки и скидки
coop - кооп-игра для 3-8 участников
achievements - достижения
rank - мой ранг
top - рейтинг чата
prozharka - случайный участник получает по заслугам
help - помощь
```

---

### 2. Get a Groq API key

1. Sign up at [console.groq.com](https://console.groq.com) — free, no credit card.
2. Go to **API Keys** → **Create API Key**.
3. Save the key as `GROQ_API_KEY`.

The main agent uses `meta-llama/llama-4-scout-17b-16e-instruct` with automatic fallback to `qwen/qwen3-32b` and `openai/gpt-oss-20b` when daily quotas are exhausted. Keyword-triggered replies use `llama-3.1-8b-instant`. Прожарки use `llama-3.3-70b-versatile`. Voice transcription uses `whisper-large-v3-turbo`.

---

### 3. Get Twitch credentials for IGDB

IGDB (game database) is owned by Twitch and uses Twitch OAuth to authenticate API requests.

1. Go to [dev.twitch.tv/console](https://dev.twitch.tv/console) and log in (or create a free account).
2. Click **Register Your Application**.
3. Fill in the form:
   - **Name:** anything (e.g. `games-chatbot-igdb`)
   - **OAuth Redirect URLs:** `http://localhost`
   - **Category:** Application Integration
4. Click **Manage** → **New Secret**.
5. Save **Client ID** as `TWITCH_CLIENT_ID` and the secret as `TWITCH_CLIENT_SECRET`.

---

### 4. Install Docker on the VPS

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

---

### 5. Clone the repo and configure secrets

```bash
git clone <repo-url>
cd games-chatbot
cp .env.example .env
nano .env
```

```env
TELEGRAM_TOKEN=1234567890:AAF...
GROQ_API_KEY=gsk_...
TWITCH_CLIENT_ID=abc123...
TWITCH_CLIENT_SECRET=xyz789...
BOT_USERNAME=@MyGamesBot
SQLITE_DB_PATH=data/chat_history.db
MAX_HISTORY_MESSAGES=10
```

```bash
chmod 600 .env
```

---

### 6. Build and start

```bash
docker compose up -d --build
```

Verify:

```bash
docker compose logs -f
```

Look for:

```
Bot started, all tables and jobs initialized
```

---

### 7. Updates

```bash
git pull
docker compose up -d --build
```

---

### Useful commands

```bash
docker compose logs -f          # stream live logs
docker compose ps               # check container status and uptime
docker compose restart          # restart without rebuilding
docker compose stop             # stop the container (keeps data)
docker compose down             # remove container, keep database volume
docker compose down -v          # remove everything including the database
```

### Resource usage

The compose file sets `mem_limit: 800m` and `memswap_limit: 800m`. Estimated steady-state RSS is 300–450 MB (Python process + MCP subprocess), leaving ~1.2 GB free for other services and the OS on a 2 GB VPS.

---

## Groq free-tier limits

| Model | Used for | RPM | TPM | TPD | RPD |
|---|---|---|---|---|---|
| `meta-llama/llama-4-scout-17b-16e-instruct` | Main agent (primary) | 30 | 30K | 500K | 1K |
| `qwen/qwen3-32b` | Main agent (fallback-1) | 60 | 6K | 500K | 1K |
| `openai/gpt-oss-20b` | Main agent (fallback-2) | 30 | 8K | 200K | 1K |
| `llama-3.1-8b-instant` | Keyword-triggered replies | 30 | 6K | 500K | 14.4K |
| `llama-3.3-70b-versatile` | Прожарки + daily roast | 30 | 12K | 100K | 1K |
| `whisper-large-v3-turbo` | Voice/video transcription | 20 | — | — | 2K |

Combined daily token budget across agent fallback chain: **1.2M tokens** (vs 200K with a single model).
