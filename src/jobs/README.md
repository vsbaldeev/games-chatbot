Scheduled background jobs — each registered as a daily Telegram JobQueue entry.

Job managers live in src/bot/jobs.py. Implementations live here, one file per job.

## Schedule

```
00:05 UTC   reset_model_job        agent.py        reset LLM fallback index to 0
03:00 UTC   cleanup_messages_job   cleanup.py      prune unified_messages and thread_history rows older than 60 days
10:00 UTC   silence_sweep_job      achievements.py award silence achievements (7/14/30 days inactive)
12:00 UTC   weekly_roast_job       roast.py        roast one random chat member (one day per week)
14:00 UTC   weekly_roles_job       roles.py        assign member title tags (Sundays only)
```

## Roast day selection

```python
# Stateless — survives restarts, varies per ISO week
year, week, _ = datetime.date.today().isocalendar()
roast_day = random.Random(year * 1000 + week).randint(0, 6)
```

## Roles job

```
1. Fetch all chat members from chat_members table
2. Load user_memories facts for each member
3. Anonymise: user_0, user_1, … (real usernames never sent to LLM)
4. LLM (llama-3.3-70b-versatile) identifies most defining trait, generates short role tag per anonymised key
5. Remap anon keys back to real user_ids
6. bot.set_chat_member_tag(chat_id, user_id, tag)   — requires can_manage_tags right
7. Announce assigned tags in chat
```
