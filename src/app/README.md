# Concepts: app, handlers, jobs

**`app` (`Application`)** — a class from the `python-telegram-bot` library, one instance per
process. It holds the bot's Telegram client, the registry that handlers get added to, and
`app.job_queue` — PTB's own `JobQueue`, backed by APScheduler internally — that jobs get added
to. `app.main()` builds the one instance this whole process uses.

**Handler** — a `(filter, callback)` pair registered with `app.add_handler(...)`. The filter
(`filters.TEXT`, `filters.PHOTO`, ...) decides whether it matches a given Update. Handlers also
have a `group` number, default `0` — PTB runs at most one matching handler per group, so a
handler in a different group (only `register_sender_as_member`, group `-1`) fires alongside the default-group
match, not instead of it.

**Job** — a `(schedule, callback)` pair registered with `app.job_queue.run_daily(...)` /
`run_once(...)`. Fires on a clock, independent of Telegram — a cron entry living inside the
same process.

**HandlerManager / JobManager** — not a PTB concept, this package's own convention: a class
that bundles several related `add_handler`/job-scheduling calls behind one
`.add_handlers(app)` / `.add_jobs(app)` method, so `app.main()` stays a short loop instead of
one long flat list of registrations.

# Wiring diagram

```
                            app.main()
                                │
                        builds Application
                                │
                                ▼
                               app
                 ┌───────────────┴───────────────┐
                 │                               │
          HandlerManagers                  JobManagers
          .add_handlers(app)                .add_jobs(app)
                 │                               │
                 ▼                               ▼
        app's handler registry             app.job_queue
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                        app.run_polling()
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      Telegram long-poll loop            job_queue clock
      ───────────────────────            ────────────────
      Update arrives                     scheduled time hits
                 │                               │
                 ▼                               ▼
      first filter match wins            matching job fires
      → runs its callback                → runs its callback
```

Both loops run inside `app.run_polling()` and share the same `app` — same bot client, same DB
pool — but fire on different triggers: one per incoming Update, one per clock tick.

`__on_startup` (registered as `post_init`) runs once, before either loop starts: it opens the DB
pool and initialises the LLM agents / TTS service, in that order — the agents query the DB, so
it must be ready first. `__on_error` wraps every handler callback and logs whatever it raises,
instead of letting PTB report "no error handlers".

## Handler managers (handlers.py)

Each class implements `HandlerManagerInterface.add_handlers(app)`. `EventHandlerManager` runs
first and registers `register_sender_as_member` in group `-1` so every update is seen before the normal
handler groups run.

### EventHandlerManager

Registers first, in group `-1`, so `register_sender_as_member` sees every update before the default-group
handlers run.

| Update / trigger | Handler | Group | Notes |
|---|---|---|---|
| any `Update` | `register_sender_as_member` | `-1` | registers every active user in `chat_members`; no chat-type filter |
| new chat member, groups only | `register_users_from_join_message` | `0` | registers each joined user in `chat_members` |

### CommandHandlerManager

| Update / trigger | Handler | Group | Notes |
|---|---|---|---|
| `/help`, groups only | `general.cmd_help` | `0` | command list |
| `/duel`, groups only | `games.cmd_duel` | `0` | emoji duel picker |
| callback query, `duel_*` pattern | `games.handle_duel_callback` | `0` | duel inline buttons |

### MessageHandlerManager

| Update / trigger | Handler | Group | Notes |
|---|---|---|---|
| text (not a command), groups only | `handle_text_message` | `0` | main pipeline entry point |
| voice or video note, groups only | `handle_voice_message` | `0` | same handler for both media types |
| photo, groups only | `handle_photo_message` | `0` | |
| sticker, groups only | `handle_sticker_message` | `0` | |
| video, groups only | `handle_video_message` | `0` | |
| animation (GIF), groups only | `handle_animation_message` | `0` | |

### Ignored updates

Not registered by any manager above — PTB drops these silently, no error.

| Update / trigger | Why ignored |
|---|---|
| any command/text/media in a private or channel chat | `group_only` (`filters.ChatType.GROUPS`) excludes `PRIVATE`/`CHANNEL`; only `register_sender_as_member` still fires there, since it has no chat-type filter |
| message reaction (add/remove emoji) | no handler registered for `message_reaction` updates anywhere in the codebase |
| audio (music/sound file attachment) | `handle_audio_message` was removed — never transcribed, no stat, rarely sent in practice |
| document, location, contact, poll, dice, venue | no filter registered for these types anywhere in the codebase |

## Scheduled jobs (jobs.py)

Each class implements `JobManagerInterface.add_jobs(app)` and registers on `app.job_queue`
(APScheduler under the hood).

### RolesJobManager

| Trigger | Job | Notes |
|---|---|---|
| daily 14:00 UTC | `weekly_roles_job` | exits early unless the day is Sunday |
| once, 30s after startup | `catch_up_roles_job` | recovers a missed Sunday run (e.g. the bot was down at 14:00 UTC) by comparing the newest stored tag timestamp against the last scheduled Sunday run; no-ops if already up to date |

### MemeJobManager

| Trigger | Job | Notes |
|---|---|---|
| daily 15:00 UTC | `daily_meme_job` | sends one fresh meme per chat |

### MessageCleanupJobManager

| Trigger | Job | Notes |
|---|---|---|
| daily 03:00 UTC | `cleanup_messages_job` | prunes `unified_messages` and `thread_history` (60-day retention) and `user_memories` facts (14-day retention) |

### YtdlpUpdateJobManager

| Trigger | Job | Notes |
|---|---|---|
| daily 03:30 UTC | `ytdlp_update_job` | installs newer yt-dlp into `/app/runtime-deps` and restarts the bot gracefully; scheduled after the cleanup job, in the chat's dead hours |

## Where the logic actually lives

`handlers.py` and `jobs.py` only wire callbacks — the callbacks themselves live in sibling
packages, each with its own README:

```
src/commands/   /help, /duel and other slash-command handlers
src/events/     handle_text_message and friends — the per-media-type update handlers
src/jobs/       the scheduled job bodies (weekly_roles_job, cleanup_messages_job, ...)
src/agent/      worker/response LLM agents, initialised in __on_startup
src/store/      Postgres access — db.init() opens the pool __on_startup depends on
src/tts/        voice reply synthesis, initialised in __on_startup
```

To trace a feature end to end: find its entry in the tables above, then open the matching file
in `src/events/` or `src/commands/`.
