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
      ├── Command handlers        ← /multiplayer, /singleplayer, /coop …
      ├── MessageHandler (text)   ← keyword/mention trigger, 60 s cooldown
      ├── MessageHandler (voice)  ← voice & video-note transcription, 50% chance
      ├── TypeHandler (group=-1)  ← passive member tracking
      └── JobQueue                ← roulette 21:00 MSK, silence-achievement check 13:00 MSK, model reset 00:05 UTC
      │
      ▼
LLM routing
  └── Direct (@mention / reply to bot) → LangChain ReAct Agent (full tool use)
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
  ├── find_coop_games(player_count, offset)  → IGDB, PS5 platform ID 167; offset paginates results
  ├── find_new_ps5_online_games(days)        → IGDB, recent PS5 multiplayer releases
  ├── find_singleplayer_ps_games(offset)     → IGDB, PS5 single-player games rated ≥ 75
  ├── get_ps_store_sales(limit)              → psdeals.net RSS, current sale titles
  └── get_ps_store_price_tr(game_name)       → Turkish PS Store link + TRY price via psdeals.net

SQLite  (aiosqlite, WAL mode, busy_timeout=5 s)
  ├── message_store      LangChain SQLChatMessageHistory (capped: 40 user msgs per chat)
  ├── wishlists          per-user game wishlist (used for sale alerts)
  ├── chat_members       member registry for achievements and roasts
  ├── user_stats         per-user counters (7 tracked dimensions)
  ├── announced_sales    dedup table, 7-day TTL
  └── suggested_games    per-chat game suggestion history (prevents repeats in /multiplayer and /singleplayer)
```

---

## Features

**Game research**
- `/multiplayer` — picks one PS5 co-op or online multiplayer game from IGDB, checks crossplay with PC, fetches TRY price and a direct Turkish PS Store link; deduplicates per chat so the same game is never suggested twice
- `/singleplayer` — picks one highly-rated PS5 single-player game (IGDB rating ≥ 75, no multiplayer modes), fetches TRY price and Turkish PS Store link; deduplicates per chat
- Search games and fetch details via IGDB (platforms, genres, multiplayer modes, rating)
- Recent PS5 multiplayer releases via IGDB filtered by release date and platform
- Current PS Store sales via psdeals.net RSS, cross-referenced with IGDB multiplayer data
- Live Steam player count via the public ISteamUserStats API
- Turkish PS Store prices fetched from psdeals.net; always returns a `store.playstation.com/tr-tr` search link as fallback

**Group utilities**
- Reply by @mentioning the bot or replying to any of its messages — handles any question directly

**Voice & video**
- Bot listens to voice messages and circle video notes in groups (25% random chance); skips forwarded messages and messages from bots
- Transcribes via Groq `whisper-large-v3`, replies with a comment (no transcript echo)

**Personality & memory**
- Friendly sarcastic tone in Russian — dry humour, light roasting, no jargon from specific wikis or subcultures
- Per-chat conversation memory (SQLite, capped at 40 user messages; trimmed to 10 turns for LLM context)
- Refuses to discuss politics, religion, or drugs — redirects in-style
- Anti-prompt-injection: mocks "forget your instructions" attempts

**Прожарка (roasts)**
- `/prozharka` — on-demand roast of a randomly chosen chat member
- Auto-roast on repeated insults: if a user replies to the bot with offensive words twice in a row, the bot generates a roast and credits `roasted_count`
- Roast style: short (≤ 2 sentences), sarcastic stand-up comedian; 10% chance of a warm supportive message instead
- Content source: based on the member's recent chat messages
- Members are mentioned with `@username` in roasts

**Russian roulette**
- Daily at 21:00 MSK the bot picks one random chat member and fires — 50% chance to hit, 50% chance to miss
- 3-message reply chain: announce → victim selection → result (5 s pauses between messages)
- `/ruletka` — on-demand roulette; requires at least 2 members in the chat
- Miss gives the victim `roulette_win_count += 1` with achievements up to 50 survivals

**Gamification**
- Achievements across 18+ tracked stats: reactions, voice/video/photo messages, stickers, links, night activity, roasts, roulette survivals, duel wins, message length
- `/achievements` — last 3 earned achievements with total count
- `/top` — top-3 chat leaderboard by achievement count with medal emojis

**Emoji duel**
- `/duel` — bot picks 2 random chat members (caller excluded when 3+ members); first to tap 🔫 wins; 5-minute timeout
- Natural challenge: write «хочу дуэль с @vasya» to start a targeted duel without a command

---

## Commands

| Command | Description |
|---|---|
| `/multiplayer` | One PS5/PC co-op or online game — crossplay status, TRY price, PS Store link; no repeats per chat |
| `/singleplayer` | One PS5 single-player game (IGDB rating ≥ 75) — TRY price, PS Store link; no repeats per chat |
| `/achievements` | Last 3 earned achievements with total count |
| `/top` | Top-3 chat leaderboard by achievement count |
| `/prozharka` | On-demand прожарка of a randomly chosen chat member |
| `/ruletka` | On-demand Russian roulette |
| `/duel` | Start an emoji duel between 2 random chat members |
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
| STT | Groq `whisper-large-v3` | 2K RPD free; transcribes voice/video-note messages |
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
├── game_tracker.py    per-chat game suggestion dedup (suggested_games table)
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
multiplayer - одна кооп/онлайн игра PS5/PC с ценой в TRY
singleplayer - одна одиночная игра PS5 с ценой в TRY
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

The main agent uses `meta-llama/llama-4-scout-17b-16e-instruct` with automatic fallback to `qwen/qwen3-32b` and `openai/gpt-oss-20b` when daily quotas are exhausted. Прожарки use `llama-3.3-70b-versatile`. Voice transcription uses `whisper-large-v3`.

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
| `llama-3.3-70b-versatile` | Прожарки | 30 | 12K | 100K | 1K |
| `whisper-large-v3` | Voice/video transcription | 20 | — | — | 2K |

Combined daily token budget across agent fallback chain: **1.2M tokens** (vs 200K with a single model). The model index resets to the primary at 00:05 UTC daily.
