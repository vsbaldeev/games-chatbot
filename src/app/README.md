# Concepts: app, handlers, jobs

**`app` (`Application`)** — a class from the `python-telegram-bot` library, one instance per
process. It holds the bot's Telegram client, the registry that handlers get added to, and
`app.job_queue` — PTB's own `JobQueue`, backed by APScheduler internally — that jobs get added
to. `app.main()` builds the one instance this whole process uses.

**Handler** — a `(filter, callback)` pair registered with `app.add_handler(...)`. The filter
(`filters.TEXT`, `filters.PHOTO`, ...) decides whether it matches a given Update. Handlers also
have a `group` number, default `0` — PTB runs at most one matching handler per group, so a
handler in a different group (only `track_member`, group `-1`) fires alongside the default-group
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
first and registers `track_member` in group `-1` so every update is seen before the normal
handler groups run.

```
EventHandlerManager
    TypeHandler(Update, track_member)           — register every active user in chat_members
    MessageHandler(new_chat_members)            — greet new members
    ChatMemberHandler(bot_added)                — handle bot being added to a new group
    MessageReactionHandler(handle_reaction)     — track emoji reactions → user_stats

CommandHandlerManager
    /start          — welcome message
    /help           — command list
    /duel           — emoji duel picker
    CallbackQueryHandler(duel_*)   — duel inline buttons

MessageHandlerManager
    text        → handle_message        — main pipeline entry point
    voice       → handle_voice_message
    video_note  → handle_voice_message  (same handler, different media_type)
    photo       → handle_photo_message
    sticker     → handle_sticker_message
    video       → handle_video_message
    animation   → handle_animation_message
    audio       → handle_audio_message
```

## Scheduled jobs (jobs.py)

Each class implements `JobManagerInterface.add_jobs(app)` and registers on `app.job_queue`
(APScheduler under the hood).

```
RolesJobManager          daily 14:00 UTC   weekly_roles_job        (exits early unless Sunday)
                          + run_once catch-up on startup, for a Sunday run missed while down
MemeJobManager           daily 15:00 UTC   daily_meme_job          (sends one fresh meme per chat)
ResetModelJobManager     daily 00:05 UTC   reset_model_job         (resets LLM fallback index to 0)
MessageCleanupJobManager daily 03:00 UTC   cleanup_messages_job    (prunes unified_messages and thread_history, 60-day retention)
YtdlpUpdateJobManager    daily 03:30 UTC   ytdlp_update_job        (installs newer yt-dlp into /app/runtime-deps and restarts the bot gracefully)
```

## Where the logic actually lives

`handlers.py` and `jobs.py` only wire callbacks — the callbacks themselves live in sibling
packages, each with its own README:

```
src/commands/   /start, /help, /duel and other slash-command handlers
src/events/     handle_message and friends — the per-media-type update handlers
src/jobs/       the scheduled job bodies (weekly_roles_job, cleanup_messages_job, ...)
src/agent/      worker/response/roast LLM agents, initialised in __on_startup
src/store/      Postgres access — db.init() opens the pool __on_startup depends on
src/tts/        voice reply synthesis, initialised in __on_startup
```

To trace a feature end to end: find its entry in the tables above, then open the matching file
in `src/events/` or `src/commands/`.
