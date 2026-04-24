# Games Chatbot

A sarcastic Telegram group chat bot for a Russian-speaking PS5 friend group. Built with a LangChain ReAct agent, a custom MCP tool server, and SQLite — deployed on a personal VPS within tight resource constraints.

The bot responds in the style of [Lurkmore](https://lurkmore.to) (a Russian satirical wiki), tracks per-user statistics, maintains conversation memory across sessions, and autonomously reacts to game-related messages.

---

## Architecture

```
Telegram Group
      │
      ▼
python-telegram-bot v22          ← polling, JobQueue, TypeHandler
      │
      ├── Command handlers        ← /crossplay, /coop, /wish, /play …
      ├── MessageHandler          ← keyword/mention trigger, 60 s cooldown
      ├── TypeHandler (group=-1)  ← passive member tracking
      └── JobQueue                ← daily roast 06:00 UTC, sale check 07:00 UTC
      │
      ▼
LangChain ReAct Agent
  model : openai/gpt-oss-20b via Groq
  memory: RunnableWithMessageHistory (per chat_id, last 10 turns)
  retries: exponential back-off on 429, one-shot reinit on MCP crash
      │
      ▼
MCP stdio subprocess  (src/mcp_server.py)
  ├── search_games(query)            → IGDB Apicalypse API
  ├── get_game_details(game_id)      → IGDB (multiplayer_modes, platforms)
  ├── get_steam_player_count(name)   → Steam Store + ISteamUserStats
  └── find_coop_games(player_count)  → IGDB, PS5 platform ID 167

SQLite  (aiosqlite, WAL mode, busy_timeout=5 s)
  ├── message_store      LangChain SQLChatMessageHistory
  ├── wishlists          per-user game wishlist
  ├── chat_members       member registry for achievements
  ├── user_stats         per-user counters (7 tracked dimensions)
  ├── announced_sales    dedup table, 7-day TTL
  ├── feature_requests   pending feature queue
  └── game_filters       per-user banned/known game lists
```

---

## Features

**Game research**
- Search games and fetch details via IGDB (platforms, genres, multiplayer modes, rating)
- Live Steam player count via the public ISteamUserStats API
- PS5 co-op finder: queries IGDB for games with `onlinecoopmax >= N` on platform 167
- Honest crossplay reporting — IGDB has no explicit crossplay field, bot says so rather than guessing

**Group utilities**
- `/play [HH:MM] [game]` — creates a Telegram poll, schedules a reminder in Moscow time
- `/wish` — per-user wishlist with add / list / remove / all-members view
- Daily PS Store sale scan via [psdeals.net](https://psdeals.net) RSS; notifies chats when wishlist games are discounted (7-day deduplication prevents re-spam)
- `/explain` — explains technical terms (ray tracing, DLSS, VRR…) in plain language

**Recommendation filters**
- `/ban <game>` — permanently excludes a game from suggestions for that user
- `/known <game>` — marks a game as already known/played; excluded from generic recommendations
- `/unban <game>` — removes either kind of filter
- `/myfilters` — lists active filters
- Filter hints are injected into each LLM call but not persisted to conversation history, so they don't accumulate tokens over time

**Personality & memory**
- Lurkmore-style sarcastic responses in Russian — encyclopaedic cynicism, gaming slang, mock footnotes
- Per-chat conversation memory (SQLite, trimmed async to avoid event-loop blocking)
- Autonomous keyword responses with a 60-second per-chat cooldown to stay within token budget
- Daily morning roast at 09:00 MSK — LLM generates a horoscope from that user's own past messages (no other users' content is sent)
- `/roast` — on-demand Lurkmore portrait of a randomly chosen chat member; the bot picks the target itself; 2-minute per-chat cooldown to limit token use

**Gamification**
- 12 achievements across 7 tracked stats: crossplay queries, tech explanations, night messages, research, co-op searches, session polls, sale notifications
- `/achievements [all]` — individual or full-chat badge board

**Meta**
- `/feature <description>` — LLM checks if already implemented, otherwise queues it; on next bot restart newly-shipped features are auto-announced to each chat
- Rate-limit and daily-quota errors surface as in-character static messages, never silent failures

**Security**
- Refuses politics and religion; redirects in-style
- Anti-prompt-injection: mocks "forget your instructions" attempts
- No chat data leakage: bot will not summarise or quote other users' history on request
- IGDB query strings sanitised (strips `"`, `;`, `\`) before interpolation

---

## Commands

| Command | Description |
|---|---|
| `/games` | Popular PS5 online games right now |
| `/crossplay <game>` | Crossplay info between PS5 and PC |
| `/players <game>` | Current Steam player count |
| `/research <query>` | Full breakdown: platforms, multiplayer, online count |
| `/coop <N>` | PS5 games with online co-op for N+ players |
| `/play [HH:MM] [game]` | Session poll + optional Moscow-time reminder |
| `/wish add\|list\|remove\|all` | Manage personal game wishlist |
| `/explain <term>` | Tech term explained in plain language |
| `/achievements [all]` | Sarcastic badge board |
| `/feature <description>` | Submit a feature request (30 s per-user cooldown) |
| `/features` | List pending feature requests for this chat |
| `/roast` | On-demand Lurkmore-style roast of a randomly chosen chat member |
| `/ban <game>` | Never suggest this game to me |
| `/known <game>` | I already play this — skip in generic recommendations |
| `/unban <game>` | Remove a filter |
| `/myfilters` | Show active filters |
| `/help` | Command list |

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Async ecosystem, aiosqlite, type hints |
| Telegram | python-telegram-bot v22 | JobQueue, TypeHandler, native async |
| LLM | Groq `openai/gpt-oss-20b` | 200 K tokens/day free tier; best TPD among free Groq models |
| Agent | LangChain ReAct + AgentExecutor | Tool-use loop with bounded iterations |
| Tool protocol | MCP (stdio) via langchain-mcp-adapters | Clean isolation; tool server restarts independently |
| Game data | IGDB API (Twitch OAuth) | Structured multiplayer/platform metadata |
| Player data | Steam public API | No auth required |
| Sale data | psdeals.net RSS | Free, covers all regions |
| Storage | SQLite + aiosqlite | Zero-dep, WAL mode for concurrent readers |
| Hosting | VPS (systemd) | 4 vCPU / 2 GB RAM, personal server |

---

## Key engineering decisions

**Agent built once, not per request.**
`ChatGroq`, the prompt template, `AgentExecutor`, and `RunnableWithMessageHistory` are constructed in `init_agent()` at startup and reused. This avoids repeated allocation of LangChain's graph structures on every message — important on a 2 GB VPS.

**MCP subprocess crash recovery.**
If the tool server dies (OOM, network crash), `run_agent` catches `BrokenPipeError` / `EOFError`, calls `init_agent()` to respawn the subprocess, and retries once — without requiring a full bot restart.

**Sync SQLAlchemy wrapped in `asyncio.to_thread`.**
`SQLChatMessageHistory` uses blocking SQLAlchemy. The `trim_history` function (clear + re-insert last N) runs entirely inside a thread-pool worker so the event loop is never blocked.

**Token budget engineering.**
Free Groq tier: 200 K tokens/day. At ~6–8 K tokens per agent turn (system prompt, 10-turn history, tool outputs, ReAct scratchpad, 512-token output), usable capacity is roughly 20–25 turns/day at 80% budget. Mitigations: keyword cooldown (1 auto-response/min per chat), `max_iterations=5`, `max_tokens=512`, history trimmed to 10 messages.

**SQLite WAL mode across all tables.**
Two different libraries (`aiosqlite` and SQLAlchemy from LangChain) share one database file. WAL mode lets readers run concurrently with the writer and eliminates the busy-lock contention that default journal mode causes on a loaded single-file DB.

---

## Project structure

```
src/
├── config.py          env var loading and validation (fails fast at startup)
├── mcp_server.py      standalone MCP server — IGDB + Steam tools, stdio transport
├── agent.py           LangChain ReAct agent, retry logic, MCP crash recovery
├── memory.py          SQLChatMessageHistory wrapper, async trim helper
├── bot.py             all Telegram handlers, jobs, startup
├── wishlist.py        wishlist CRUD (aiosqlite)
├── psstore.py         psdeals.net RSS fetch, sale dedup, announced_sales table
├── achievements.py    12 achievements, 7 tracked stats, migration helper
├── features.py        feature-request queue, LLM-based implementation checker
└── game_filters.py    per-user banned/known game filter lists (aiosqlite)
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
games - популярные игры для PS5
crossplay - кросплей <игра>
players - онлайн игроков <игра>
research - анализ <запрос>
coop - кооп на N игроков
play - опрос кто играет сегодня
wish - вишлист игр
explain - объяснить термин
achievements - достижения
feature - предложить фичу
features - список запросов фич
roast - луркморский портрет участника
ban - никогда не предлагать <игра>
known - я уже знаю эту игру <игра>
unban - убрать из фильтров <игра>
myfilters - мои фильтры рекомендаций
help - помощь
```

---

### 2. Get a Groq API key

1. Sign up at [console.groq.com](https://console.groq.com) — free, no credit card.
2. Go to **API Keys** → **Create API Key**.
3. Save the key as `GROQ_API_KEY`.

The bot uses model `openai/gpt-oss-20b`. The free tier gives 200 000 tokens/day and 30 requests/minute.

---

### 3. Get Twitch credentials for IGDB

IGDB (game database) is owned by Twitch and uses Twitch OAuth to authenticate API requests. You never actually use Twitch itself — the credentials are only for obtaining short-lived IGDB access tokens.

1. Go to [dev.twitch.tv/console](https://dev.twitch.tv/console) and log in (or create a free account).
2. Click **Register Your Application**.
3. Fill in the form:
   - **Name:** anything (e.g. `games-chatbot-igdb`)
   - **OAuth Redirect URLs:** `http://localhost` (not actually used, but required)
   - **Category:** Application Integration
4. Click **Manage** on the newly created app → **New Secret**.
5. Save **Client ID** as `TWITCH_CLIENT_ID` and the generated secret as `TWITCH_CLIENT_SECRET`.

The bot exchanges these credentials for a bearer token at startup and automatically refreshes it before expiry.

---

### 4. Install Docker on the VPS

If Docker is already installed on your VPS, skip this step.

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # run docker without sudo after re-login
newgrp docker                   # apply group change in the current session
```

---

### 5. Clone the repo and configure secrets

```bash
git clone <repo-url>
cd games-chatbot
cp .env.example .env
nano .env
```

Fill in all values. Leave `SQLITE_DB_PATH` as-is — Docker Compose overrides it to point at the named volume automatically.

```env
TELEGRAM_TOKEN=1234567890:AAF...
GROQ_API_KEY=gsk_...
TWITCH_CLIENT_ID=abc123...
TWITCH_CLIENT_SECRET=xyz789...
BOT_USERNAME=@MyGamesBot
SQLITE_DB_PATH=data/chat_history.db   # overridden by compose, leave it
MAX_HISTORY_MESSAGES=10
```

Lock down the file so only your user can read it:

```bash
chmod 600 .env
```

---

### 6. Build and start

```bash
docker compose up -d --build
```

This builds the image, creates a named Docker volume for the SQLite database, and starts the container in the background. The database persists across rebuilds and restarts.

Verify it started cleanly:

```bash
docker compose logs -f
```

Look for this line in the output:

```
Bot started, all tables and jobs initialized
```

That confirms the Telegram connection, Groq API, IGDB token exchange, and SQLite setup all succeeded. Add the bot to your group and send `/help`.

---

### 7. Updates

```bash
git pull
docker compose up -d --build
```

Docker rebuilds the image from the new code, replaces the container, and keeps the database volume untouched. On the first startup after an update the bot automatically announces any newly implemented feature requests to each chat.

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

## Groq free-tier limits (`openai/gpt-oss-20b`)

| Limit | Value |
|---|---|
| Requests per minute | 30 |
| Tokens per day | 200 000 |
| Requests per day | 1 000 |

The bot uses ~80% of the daily token budget as a soft ceiling. Rate-limit hits (HTTP 429) are retried with exponential back-off (5 s → 10 s → 20 s). Daily quota exhaustion surfaces as an in-character static message; the bot recovers automatically the next day.
