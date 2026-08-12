Telegram group chat bot for a Russian-speaking PS5 friend group.

The bot is НейроЖора — Жора for short — a sarcastic gaming assistant written as a
do-it-all guy from an unnamed Slavic village: he answers game questions with dry
humour, needles the members, and decides on his own when a joke is worth making
(gentle by default, roasting only when someone invites it) or when to stay quiet.

The village is his manner, not his subject. Жора used to post episodes from an
invented village life three times a week and to weave that biography into ordinary
replies; the chat read it as noise, so the whole feature was retired on 2026-08-10
and he now speaks only when spoken to (see `src/life/README.md` for the decision
and what was removed). What survives is the voice — and the photos.

Members can ask Жора for a photo of himself («сфоткай себя», «покажи свой
огород»): the filter classifies the request, Жора immediately answers in
character that he went to take the shot, and the generated photo arrives as a
reply minutes later (CPU generation is slow). Photo requests are the heaviest
signal of the per-user attention budget — a rapid second request gets an
in-character refusal instead of a second render — and a single global
generation slot keeps concurrent requests from queueing behind each other.
Between posts, a silent daily job invents a new season-appropriate activity
phrase each morning with no chat post, so the answer changes daily instead of
sitting frozen for up to a week — and a dated history of recent activities
lets Жора answer "what did you do yesterday" consistently too.
Tracks per-user stats, extracts long-term memories, and routes every message through
a typed LangGraph pipeline. YouTube Shorts, Instagram Reels, and long-form YouTube
links posted in the chat are watched for everyone: the bot downloads the clip
(Shorts and Reels only — long-form YouTube never carries video), transcribes and
looks at it (Shorts) or reads its title/caption and top comments (Reels and
long-form YouTube), and replies with a single message — the video with a 1–2
sentence summary and comment-reaction recap as its caption, or, when there's no
video, a plain text message with a link preview. If the sender's message was
nothing but the link, the bot deletes it once its own message has gone out and
credits the sender inside the caption ("Скинул @username" + the link + the
summary); if the sender wrote anything else alongside the link, their message is
left untouched and the bot's message replies to it instead, carrying just the
summary. A failed combined send falls back to an ordinary text reply, and the
original is never deleted when that happens. Deleting the original requires the
bot to be a chat administrator with the `can_delete_messages` permission — without
it, everything else still works and the link message simply stays in the chat.
Replying directly to the bot's link-repost message with a real question gets
a grounded answer, using the same transcript/comment material the caption
was written from — but a bare reply with no question and no `@mention`
doesn't automatically count as addressing the bot the way replying to its
other messages does, since it's often just chat among members about the
video rather than talk to the bot.
Voice messages and
video notes are answered in kind: the reply comes back as a voice note spoken
by a local Silero v5 Russian TTS voice, degrading to plain text whenever the
reply is unspeakable (too long, no Cyrillic) or synthesis fails.

## Architecture

| Module | Description |
|---|---|
| [src/pipeline/](src/pipeline/README.md) | LangGraph StateGraph — message processing nodes and graph wiring |
| [src/bot/](src/bot/README.md) | Application wiring — handler registration, job setup, startup lifecycle |
| [src/events/](src/events/README.md) | Telegram event handlers — member tracking, reactions, messages |
| [src/commands/](src/commands/README.md) | Command handlers — /duel |
| [src/jobs/](src/jobs/README.md) | Scheduled jobs — weekly roles, daily meme, cleanup, yt-dlp refresh |
| [src/life/](src/life/README.md) | Chat-requested selfies — scene writing, best-of-N generation, vision judge |
| [src/tools/](src/tools/README.md) | MCP tool server — IGDB, Steam, PS Store, TMDB, AniList, web search |
| [src/store/](src/store/README.md) | asyncpg data access — messages, memories, thread history, embeddings |
| [src/achievements/](src/achievements/README.md) | Duel achievements + stat counters (consumed by roast material) |
| [src/tts/](src/tts/README.md) | Text-to-speech — local Silero v5 Russian synthesis for voice-in-kind replies |
| [src/imagegen/](src/imagegen/README.md) | HTTP client for the imagegen service — never-raise, photo posts degrade to text |
| [imagegen-service/](imagegen-service/README.md) | Self-hosted CPU image generation — FastAPI + diffusers + LCM-LoRA, separate container |
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
LLM (roles)  Groq llama-3.3-70b-versatile
LLM (filter) Groq llama-3.3-70b-versatile → OpenRouter meta-llama/llama-3.3-70b-instruct (cross-provider fallback; the 8B model dropped real questions)
STT          Groq whisper-large-v3
TTS          Silero v5 Russian (local, CPU torch, speaker aidar; OGG/Opus via PyAV)
Image gen    Stable Diffusion 1.5 (DreamShaper 8), DPM++ 2M Karras 20 steps, self-hosted CPU service (diffusers/FastAPI, async job API); best-of-3 candidates ranked by the Groq vision judge
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

### Logging

Log lines follow a compact aligned format — `DD.MM HH:MM:SS L corr logger message`,
where `L` is the one-character level and `corr` is a per-message correlation id
derived from the Telegram update id. Every handled message produces exactly one
**canonical INFO line** (logger `pipeline`) summarizing the whole run:

```
14.07 21:03:12 I 842137 pipeline  chat=-100123 user=@vasya kind=voice msg=5121 trigger=explicit filter=MEANINGFUL tier=0 guard=ok action=replied len=214 dur=3.42s
```

`action` is `replied`, `joked`, `replied+photo` (a photo-request ack whose
background selfie generation was launched), `ignored` (with `reason=…`:
`not_addressed`, `meaningless`, `wound_down`, `guard_blocked`, …) or
`error:<kind>`.

Policy: INFO carries metadata only — no message text, transcripts or reply bodies.
Content excerpts appear only at DEBUG, truncated. Per-node decision traces
(filter/guard/engagement) are DEBUG and share the canonical line's correlation id.
One deliberate exception: the response node dumps the exact input the reply LLM
sees (`Response LLM final turn` — verbatim and multi-line, since that assembled
prompt exists nowhere else; thread-history turns as one-line excerpts) at DEBUG,
for diagnosing absurd replies.

Env knobs (set on the bot service in `docker-compose.yml`):

- `LOG_LEVEL` — root level, default `INFO`; set `DEBUG` to see per-node traces.
- `LOG_COLOR` — `always` / `never` / `auto` (tty detection). The compose file sets
  `always`; note `docker compose logs --no-color` does not strip app-emitted ANSI,
  so set `LOG_COLOR: never` if logs are shipped to a plain-text collector.
- `ASYNCPG_LOG_LEVEL`, `TELEGRAM_APP_LOG_LEVEL`, `TELEGRAM_UPDATER_LOG_LEVEL` —
  per-library overrides (default WARNING in compose). `httpx`, `httpcore`,
  `telegram.ext.ExtBot` and `apscheduler` are always muted to WARNING.
- `GROQ_LOG_LEVEL` — default `WARNING`. The Groq SDK logs whole request bodies at
  DEBUG, and for the vision judge and the photo/sticker describers those bodies
  embed a base64-encoded PNG — hundreds of KB of noise per call, which buries
  everything else at `LOG_LEVEL=DEBUG`. Raise it to `DEBUG` only when debugging
  the HTTP layer itself.

#### Reading back what the bot said

What the bot actually says is logged by the bot, not by the SDK. Every outgoing
text shares the logger name `outgoing`, so one grep is a transcript:

```
docker compose logs bot | grep outgoing
14.07 21:03:12 D 842137 outgoing         reply chat=-100123 Да не, я такое не сажаю…
14.07 17:00:41 D -      outgoing         life:photo chat=-100123 Красил забор, вышло криво…
```

The first field is what went out — `reply`, `joke`, or `life:<format>` for a
scheduled post. Voice posts log both halves (`[caption] … [spoken] …`), since the
caption alone doesn't tell you what was said. These are DEBUG records, per the
policy above; the canonical INFO line still carries only `len=`.

The imagegen sidecar keeps uvicorn's own logging and is not affected.

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
and **admin rights** with `can_manage_tags` for weekly member roles and
`can_delete_messages` for the single-message link repost.

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
help - помощь
```
