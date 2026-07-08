Telegram group chat bot for a Russian-speaking PS5 friend group.

Friendly and sarcastic personality — answers game questions with dry humour and
decides on its own when to drop a joke into the chat (a conversation-spawning hook,
gentle by default, roasting only when someone invites it) or to stay quiet. Tracks
per-user stats, extracts long-term memories, and routes every message through a
typed LangGraph pipeline. YouTube Shorts links posted in the chat are watched for
everyone: the bot downloads the short, transcribes and looks at it, reads the top
comments, and replies with a 1–2 sentence retell of the video plus a short summary
of the audience reaction in the comments — no verdict, and no fact-checking the
video against the model's own (possibly stale) knowledge. Voice messages and
video notes are answered in kind: the reply comes back as a voice note spoken
by a local Silero v5 Russian TTS voice, degrading to plain text whenever the
reply is unspeakable (too long, no Cyrillic) or synthesis fails.

## Architecture

| Module | Description |
|---|---|
| [src/pipeline/](src/pipeline/README.md) | LangGraph StateGraph — message processing nodes and graph wiring |
| [src/bot/](src/bot/README.md) | Application wiring — handler registration, job setup, startup lifecycle |
| [src/events/](src/events/README.md) | Telegram event handlers — member tracking, reactions, messages |
| [src/commands/](src/commands/README.md) | Command handlers — /duel, /meme |
| [src/jobs/](src/jobs/README.md) | Scheduled jobs — weekly roles, daily meme, cleanup |
| [src/tools/](src/tools/README.md) | MCP tool server — IGDB, Steam, PS Store, TMDB, AniList, web search |
| [src/store/](src/store/README.md) | asyncpg data access — messages, memories, thread history, embeddings |
| [src/achievements/](src/achievements/README.md) | Duel achievements + stat counters (consumed by roast material) |
| [src/tts/](src/tts/README.md) | Text-to-speech — local Silero v5 Russian synthesis for voice-in-kind replies |
| [src/config/](src/config/) | Configuration — credentials, model registry ([models.py](src/config/models.py)), and every LLM prompt text ([prompts.py](src/config/prompts.py)) |
| [alembic/](alembic/) | Database schema migrations (Alembic) — the sole owner of the schema |

## Tech stack

```
Language     Python 3.13
Telegram     python-telegram-bot v22 (JobQueue, native async)
Pipeline     LangGraph StateGraph
Tools        MCP (stdio) via langchain-mcp-adapters
LLM (agent)  Groq gpt-oss-120b → qwen3.6-27b → gpt-oss-20b (fallback chain; no 8B floor — it fabricates instead of calling tools)
LLM (memory) Groq qwen/qwen3.6-27b (reasoning disabled — thinking would eat the whole token budget)
Embeddings   fastembed paraphrase-multilingual-MiniLM-L12-v2 (ONNX, 384-dim, local)
LLM (roast)  Groq openai/gpt-oss-120b → llama-3.3-70b-versatile → gpt-oss-20b (fallback chain)
LLM (humor)  Groq openai/gpt-oss-120b → llama-3.3-70b-versatile → qwen3.6-27b (autonomous comedian; JSON decide-or-abstain)
LLM (roles)  Groq llama-3.1-8b-instant
STT          Groq whisper-large-v3
TTS          Silero v5 Russian (local, CPU torch, speaker aidar; OGG/Opus via PyAV)
Vision       Groq qwen/qwen3.6-27b (reasoning disabled — thinking would eat the whole token budget)
Security     Groq llama-prompt-guard-2-86m
Video frames PyAV (in-process, no subprocess)
Shorts DL    yt-dlp (in-process Python API; self-updates on start + daily check)
PO tokens    bgutil-ytdlp-pot-provider sidecar (defeats YouTube bot-detection on VPS IPs)
Game data    IGDB (Twitch OAuth), Steam public API, psdeals.net RSS
Media data   TMDB, AniList GraphQL, OpenCritic
Web search   Tavily (falls back to DuckDuckGo)
Storage      PostgreSQL + pgvector + asyncpg (connection pool, min 2 / max 10)
Hosting      VPS / Docker Compose (bot + postgres + pot-provider containers)
```

## Running

```bash
# Local (requires .env from .env.example)
alembic upgrade head   # apply schema migrations first
python -m src.bot

# Production
docker compose up -d --build
docker compose logs -f
```

Required env vars: `TELEGRAM_TOKEN`, `GROQ_API_KEY`, `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`, `BOT_USERNAME`, `POSTGRES_PASSWORD`.
Optional: `DATABASE_URL` (defaults to `postgresql://chatbot:changeme@localhost:5432/chatbot`), `TAVILY_API_KEY`, `TMDB_API_KEY`,
`TTS_MODEL_PATH` (defaults to `.cache/silero/v5_ru.pt`; downloaded on first start locally, pre-baked in the Docker image).

Voice replies run on CPU-only PyTorch (`torch==2.9.1+cpu` from the PyTorch wheel
index on Linux) — the bot container's memory limit is 2g to fit the torch runtime
plus the resident Silero model. A TTS load failure is not fatal: the bot starts and
answers everything in text.

## Database migrations

The schema is owned entirely by **Alembic** — the bot no longer creates tables at
startup. In Docker, `entrypoint.sh` runs `alembic upgrade head` before the bot
process starts, so containers self-provision on every deploy. Migrations resolve
`DATABASE_URL` through `src/db_url.py`, a standalone module that pulls in no bot
credentials, so they run without a Telegram token or LLM keys.

```bash
alembic upgrade head            # apply all pending migrations
alembic revision -m "add x"     # create a new (hand-written) migration
alembic current                 # show the DB's current revision
```

Migrations are **forward-only** — this service does not support downgrades, so
`downgrade()` raises `NotImplementedError`. Roll back by writing a new forward
migration.

> **Existing database (first upgrade to this version):** the tables already exist
> (an earlier bot created them), so baseline the database instead of re-creating
> them — run `alembic stamp head` once. Fresh databases just run `alembic upgrade head`.

Schema DDL lives as raw SQL inside `alembic/versions/` because the app talks to
PostgreSQL through asyncpg with no SQLAlchemy models; autogenerate is not used.

## Deploy

```bash
# First time
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
git clone <repo-url> && cd games-chatbot
cp .env.example .env && nano .env
chmod 600 .env
docker compose up -d --build

# Updates
git pull && docker compose up -d --build
```

Bot requires **Privacy Mode off** (BotFather → Bot Settings → Group Privacy → Turn off)
and **admin rights** with `can_manage_tags` for weekly member roles.

### Shorts summaries are self-maintaining

The YouTube pieces rot on purpose (YouTube fights downloaders), so all of the
maintenance is automated: the `pot-provider` sidecar generates the PO tokens
YouTube demands from datacenter IPs, `entrypoint.sh` upgrades yt-dlp into
`/app/runtime-deps` on every container start, and a daily 03:30 UTC job installs
newer yt-dlp releases and restarts the bot gracefully. Nothing to configure — no
cookies, no extra env vars. If Shorts summaries ever go silent anyway,
`docker compose logs bot | grep -i shorts` shows which stage is failing.

## BotFather commands

```
duel - эмодзи-дуэль между двумя участниками
meme - случайный мем
help - помощь
```
