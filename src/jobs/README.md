Scheduled background jobs — each registered as a daily Telegram JobQueue entry.

Job managers live in src/bot/jobs.py. Implementations live here, one file per job.

## Schedule

```
00:05 UTC        reset_model_job        agent.py        reset LLM fallback index to 0
03:00 UTC        cleanup_messages_job   cleanup.py      prune unified_messages and thread_history rows older than 60 days
03:30 UTC        ytdlp_update_job       ytdlp_update.py install newer yt-dlp into /app/runtime-deps and restart the bot gracefully (SIGTERM + docker restart policy); no-op outside the container or when current
09:30 MSK        daily_activity_job     daily_activity.py silently invent Жора's current-activity phrase for the day, no chat post; skipped if already refreshed today (e.g. a life post landed); see src/life/README.md
17:00 MSK        life_post_job          life_post.py    post one of Жора's life-story episodes — Mon photo, Wed voice, Sat story; other days no-op; see src/life/README.md
14:00 UTC        weekly_roles_job       roles.py        assign unique member role tags + reasons (Sundays only)
15:00 UTC        daily_meme_job         meme.py         send one fresh unseen meme to every chat (every day)
```

## Roles job

The whole pipeline is keyed by `user_id`. Display names — which fall back to a
non-unique first name — are used only when rendering the announcement, so two
members who share a name can never collapse into one entry.

```
1. Fetch all chat members (chat_members table); keep names_by_uid for rendering
2. Load user_memories facts for each member; eligible = members that have facts
   (factless members are left untagged)
3. generate_roles: anonymise to user_0, user_1, … (real ids never sent to LLM);
   LLM (llama-3.3-70b-versatile) returns {role, reason} per anon key; remap back
4. fill_missing_roles: members the LLM omitted are re-asked once, then any still
   missing get the neutral FALLBACK_ROLE + reason — every eligible member ends up tagged
5. enforce_unique_roles: case-insensitive duplicate roles trigger one re-ask for
   distinct alternatives; a deterministic suffix pass guarantees strict uniqueness
6. Persist every role + reason to the user_tags table (upsert by chat_id, user_id)
7. announce_roles: message is built from the decided role map (NOT from API success),
   so a member is never dropped from the list. Each member's role is shown together
   with its LLM-generated reason as a one-sentence profile line (render_member_block).
   The sent announcement is recorded in unified_messages so a reply to it carries the
   role list as replied-to context
8. apply_telegram_tags: best-effort bot.set_chat_member_tag per member — failures
   (e.g. Chat_creator_required) are swallowed and do not affect the announcement
```

Reasons stored in `user_tags` let the response pipeline explain a member's role
when they ask "why do I have this role?" (see `src/pipeline/README.md`).

## Daily activity job

Full generation flow lives in `src/life/README.md`. This job file only owns
scheduling: a daily 09:30 MSK run (well before the 17:00 life-post slot, so a
same-day life post always ends up as the newer
`current_activity`) that skips silently if today's activity was already set
— by an earlier run of this job or by a life post landing before 09:30.
`catch_up_daily_activity_job` recovers a refresh missed while the bot was
down, using the same "already refreshed today" check.

## Life-post job

Full episode-writing and posting flow lives in `src/life/README.md`. This job
file owns scheduling **and format**: `WEEKLY_SCHEDULE` maps weekday → format
(Mon photo, Wed voice, Sat story), all at 17:00 MSK. A daily "run at 17:00,
act only on scheduled days" trigger fires the post; catch-up recovers a
missed slot on startup and also fires the very first post ever right after
deployment.

The format used to be the episode writer's own choice among the offered
formats, constrained only by "don't repeat the previous post" — which
endless story/voice alternation satisfies forever. In production the writer
picked `photo` exactly zero times in six posts, so the whole image-generation
path never ran once. Cadence is a scheduling decision, not a creative one;
it lives here now and the writer is simply told which format to write for.

### Startup catch-up

`weekly_roles_job` only does work on Sundays, so a Sunday spent down (e.g. a
network outage at 14:00 UTC) means members go a whole week with no roles. To
recover, `RolesJobManager` also registers a one-off `catch_up_roles_job` shortly
after startup. It compares the newest `user_tags.assigned_at` against the most
recent scheduled Sunday run (`last_scheduled_roles_run`): if nothing was assigned
at or after that run, it runs the assignment once now. When the latest run is
already covered it logs and skips, so a normal restart never re-runs the job.
