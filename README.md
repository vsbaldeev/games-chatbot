Telegram group chat bot for a Russian-speaking PS5 friend group.

Friendly and sarcastic personality — jokes around, roasts chat members, answers game
questions with dry humour. Tracks per-user stats, extracts long-term memories, and
routes every message through a typed LangGraph pipeline.

## Architecture

```
src/pipeline/       LangGraph StateGraph — message processing nodes and graph wiring
src/tools/          MCP tool server — IGDB, Steam, PS Store, TMDB, AniList, web search
src/store/          asyncpg data access — unified_messages, user_memories, user_stats
src/achievements/   Achievement system — stat rules, unlock checks, announcements
src/events/         Telegram event handlers — member tracking, reactions, messages
src/commands/       Command handlers — /duel, /dnd_*, /roast, /top, /achievements
src/jobs/           Scheduled jobs — weekly roast, weekly roles, silence sweep, model reset
src/bot/            Application wiring — handler registration, job setup, startup lifecycle
```

## Forwarded posts

The bot ignores forwarded messages and does not respond to them directly.
To ask the bot about a forwarded post, reply to it and mention the bot.

When building context for such a reply the bot:
1. Walks the reply chain to the forwarded message.
2. If the forwarded post is part of a media group (album with multiple photos/videos), fetches all sibling messages by `media_group_id` and merges them into the context.
3. Enriches each media item lazily — photos are described via vision LLM, voice/video is transcribed via Whisper.
4. Passes the enriched content to the worker, which can also search the web using the described content.

## Tech stack

```
Language     Python 3.13
Telegram     python-telegram-bot v22 (JobQueue, native async)
Pipeline     LangGraph StateGraph
Tools        MCP (stdio) via langchain-mcp-adapters
LLM (agent)  Groq llama-4-scout-17b-16e-instruct → qwen3-32b → gpt-oss-20b (fallback chain)
LLM (memory) Groq meta-llama/llama-4-scout-17b-16e-instruct
Embeddings   fastembed paraphrase-multilingual-MiniLM-L12-v2 (ONNX, 384-dim, local)
LLM (roast)  Groq llama-3.3-70b-versatile
LLM (roles)  Groq llama-3.1-8b-instant
STT          Groq whisper-large-v3
Vision       Groq llama-4-scout-17b-16e-instruct
Security     Groq llama-prompt-guard-2-86m
Video frames PyAV (in-process, no subprocess)
Game data    IGDB (Twitch OAuth), Steam public API, psdeals.net RSS
Media data   TMDB, AniList GraphQL, OpenCritic
Web search   Tavily (falls back to DuckDuckGo)
Storage      PostgreSQL + pgvector + asyncpg (connection pool, min 2 / max 10)
Hosting      VPS / Docker Compose (bot + postgres containers)
```

## Running

```bash
# Local (requires .env from .env.example)
python -m src.bot

# Production
docker compose up -d --build
docker compose logs -f
```

Required env vars: `TELEGRAM_TOKEN`, `GROQ_API_KEY`, `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`, `BOT_USERNAME`, `POSTGRES_PASSWORD`.
Optional: `DATABASE_URL` (defaults to `postgresql://chatbot:changeme@localhost:5432/chatbot`), `TAVILY_API_KEY`, `TMDB_API_KEY`.

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

## BotFather commands

```
dnd_pvp - D&D приключение, 1 раунд, все против всех
dnd_coop - D&D кооп, 2 раунда против босса
dnd_heist - Великое Ограбление — 3 фазы
duel - эмодзи-дуэль между двумя участниками
roast - случайный участник получает по заслугам
achievements - последние достижения
top - топ чата по достижениям
help - помощь
```
