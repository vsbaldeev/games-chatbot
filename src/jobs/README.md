Scheduled background jobs — each registered as a daily Telegram JobQueue entry.

Job managers live in src/bot/jobs.py. Implementations live here, one file per job.

## Schedule

```
00:05 UTC        reset_model_job        agent.py        reset LLM fallback index to 0
03:00 UTC        cleanup_messages_job   cleanup.py      prune unified_messages and thread_history rows older than 60 days
03:30 UTC        ytdlp_update_job       ytdlp_update.py install newer yt-dlp into /app/runtime-deps and restart the bot gracefully (SIGTERM + docker restart policy); no-op outside the container or when current
10:00 MSK        life_post_job          life_post.py    post one of Жора's life-story episodes, on 2 random days per week at a random daytime minute (never at night); see src/life/README.md
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

## Life-post job

Full episode-writing and posting flow lives in `src/life/README.md`. This job
file only owns scheduling: a deterministic per-week random plan (seeded on
ISO year/week, no schedule table) picks 2 days and a random daytime minute
each; a daily "run at window start, act conditionally" trigger fires the
actual post; catch-up recovers a missed slot on startup and also fires the
very first post ever right after deployment.

### Startup catch-up

`weekly_roles_job` only does work on Sundays, so a Sunday spent down (e.g. a
network outage at 14:00 UTC) means members go a whole week with no roles. To
recover, `RolesJobManager` also registers a one-off `catch_up_roles_job` shortly
after startup. It compares the newest `user_tags.assigned_at` against the most
recent scheduled Sunday run (`last_scheduled_roles_run`): if nothing was assigned
at or after that run, it runs the assignment once now. When the latest run is
already covered it logs and skips, so a normal restart never re-runs the job.
