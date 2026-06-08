Scheduled background jobs — each registered as a daily Telegram JobQueue entry.

Job managers live in src/bot/jobs.py. Implementations live here, one file per job.

## Schedule

```
00:05 UTC   reset_model_job        agent.py        reset LLM fallback index to 0
03:00 UTC   cleanup_messages_job   cleanup.py      prune unified_messages and thread_history rows older than 60 days
10:00 UTC   silence_sweep_job      achievements.py award silence achievements (7/14/30 days inactive)
12:00 UTC   weekly_roast_job       roast.py        roast one random chat member (one day per week)
14:00 UTC   weekly_roles_job       roles.py        assign unique member role tags + reasons (Sundays only)
```

## Roast day selection

```python
# Stateless — survives restarts, varies per ISO week
year, week, _ = datetime.date.today().isocalendar()
roast_day = random.Random(year * 1000 + week).randint(0, 6)
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
   so a member is never dropped from the list
8. apply_telegram_tags: best-effort bot.set_chat_member_tag per member — failures
   (e.g. Chat_creator_required) are swallowed and do not affect the announcement
```

Reasons stored in `user_tags` let the response pipeline explain a member's role
when they ask "why do I have this role?" (see `src/pipeline/README.md`).
