Scheduled background jobs — each registered as a daily Telegram JobQueue entry.

Job managers live in src/bot/jobs.py. Implementations live here, one file per job.

## Schedule

```
00:05 UTC        reset_model_job        agent.py        reset LLM fallback index to 0
03:00 UTC        cleanup_messages_job   cleanup.py      prune unified_messages and thread_history rows older than 60 days
03:30 UTC        ytdlp_update_job       ytdlp_update.py install newer yt-dlp into /app/runtime-deps and restart the bot gracefully (SIGTERM + docker restart policy); no-op outside the container or when current
14:00 UTC        weekly_roles_job       roles.py        assign unique member role tags + reasons (Sundays only)
15:00 UTC        daily_meme_job         meme.py         fan out memes.sender.send_meme over every chat (un-anchored, vision-vetted, no caption)
```

## Roles job

The whole pipeline is keyed by `user_id`. Display names — which fall back to a
non-unique first name — are used only when rendering the announcement, so two
members who share a name can never collapse into one entry.

```
1. Fetch all chat members (chat_members table); keep names_by_uid for rendering
2. Load user_memories facts for each member; eligible = members that have facts
   (factless members are left untagged)
3. generate_roles: anonymise to user_0, user_1, … via src/utils/anon_map.py
   (shared with src/group_profile/, real ids never sent to LLM);
   LLM (TAG_MODEL, qwen/qwen3.6-27b) returns {role, reason} per anon key;
   remap back. Called with reasoning_effort="none", which this model accepts
   (the earlier gpt-oss-120b primary 400s on "none" and needed a "low"
   budget workaround instead — see git history); "none" leaves the whole
   max_tokens=2048 free for the JSON body rather than budgeting reasoning
   against it. Transient TPM 429s are retried with backoff
   (src.agent.ainvoke_with_backoff), since a truncated call's re-ask is a
   second full-size request in the same window. Output is passed through
   normalize_homoglyphs (src.agent.language) — this model occasionally
   splices a Latin/Greek glyph into an otherwise-Cyrillic word
4. fill_missing_roles: members the LLM omitted are re-asked once, then any still
   missing get the neutral FALLBACK_ROLE + reason — every eligible member ends up tagged
5. enforce_unique_roles: case-insensitive duplicate roles trigger one re-ask for
   distinct alternatives; a deterministic suffix pass guarantees strict uniqueness
6. Persist every role + reason to the user_tags table (upsert by chat_id, user_id)
7. announce_roles: message is built from the decided role map (NOT from API success),
   so a member is never dropped from the list. Each member's role is shown together
   with its LLM-generated reason as a one-sentence profile line (render_member_block).
   The sent announcement is recorded in unified_messages marked ``is_broadcast=True``
   so a reply to it carries the role list as replied-to context, and replies can be
   gated as chat among members rather than necessarily as messages to the bot
8. apply_telegram_tags: best-effort bot.set_chat_member_tag per member — failures
   (e.g. Chat_creator_required) are swallowed and do not affect the announcement
```

Reasons stored in `user_tags` let the response pipeline explain a member's role
when they ask "why do I have this role?" (see `src/pipeline/README.md`).

### Startup catch-up

`weekly_roles_job` only does work on Sundays, so a Sunday spent down (e.g. a
network outage at 14:00 UTC) means members go a whole week with no roles. To
recover, `RolesJobManager` also registers a one-off `catch_up_roles_job` shortly
after startup. It compares the newest `user_tags.assigned_at` against the most
recent scheduled Sunday run (`last_scheduled_roles_run`): if nothing was assigned
at or after that run, it runs the assignment once now. When the latest run is
already covered it logs and skips, so a normal restart never re-runs the job.
