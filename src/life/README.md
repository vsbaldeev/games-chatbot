Жора's scheduled life posts — the character's own story, told to every chat twice a
week and remembered as canon.

## Flow

```
src/jobs/life_post.py (schedule) ──► src/life/poster.py:post_life_episode(bot)
    │
    ├─ src/life/writer.py: EpisodeWriterAgent.write_episode()
    │      reads bot_memories.get_recent_episodes(10) + get_writer_facts()
    │      (20 newest + 10 sampled older); src/life/engagement.choose_mode()
    │      picks how this post engages the chat (see below) → prompts
    │      EPISODE_WRITER_SYSTEM → strict JSON {episode_text, image_prompt,
    │        voice_script, current_activity, format} → parse_episode
    │        validates length (≤ EPISODE_TEXT_MAX_CHARS, 450) and required
    │        fields; one retry on a malformed/invalid response, then None
    │        (slot skipped, catch-up retries later)
    │
    ├─ send_episode: fan out to every chat (achievements.get_all_chat_ids,
    │      asyncio.gather(..., return_exceptions=True) — one chat's failure
    │      cannot abort the others), record each send in unified_messages
    │      (content = episode_text, so the bot's own posts need no
    │      transcription/vision when a member replies to them)
    │
    └─ only if ≥1 chat received the post:
           record_episode → bot_memories.insert_episode(...) with the
           episode's current_activity, then distill_facts (BOT_FACT_DISTILL_SYSTEM
           + MEMORY_MODEL, reasoning disabled) → bot_memories.upsert_facts(...)
           (semantic dedup, newest text wins on a match)
```

Zero successful sends leaves `bot_memories` untouched — the watermark
(`get_latest_posted_at`) stays behind, so catch-up retries the slot on the
next startup instead of silently losing it.

## Engagement — `src/life/engagement.py`

Жора's posts don't just narrate his own life — each one either asks the chat
a question or pulls a real member into the story:

- **SOLO** (`MEMBER_MODE_CHANCE = 0.5` chance of *not* landing on MEMBER):
  the episode closes with a question or callout aimed at the chat.
- **MEMBER**: `collect_mentionable_facts()` gathers `(username, fact)` pairs
  across every registered chat from `user_memories` (in practice a single
  friend group — life posts broadcast identically everywhere, matching
  `bot_memories`' "one life" design), `is_safe_to_mention` drops counter-tally
  facts («Оскорблял бота N раз», «Пытался взломать бота N раз») and
  second-hand cross-user facts («по словам @X, ...») since broadcasting
  either would misrepresent or embarrass the member, and one surviving
  candidate is picked at random. The writer prompt is instructed to weave
  the fact in warmly and to soften or drop it if it reads as too personal —
  the prefix filter is a mechanical first pass, not a full judgment call.
  Falls back to SOLO when no eligible candidate exists (fresh install, or
  every stored fact is filtered out) or the lookup itself fails — a broken
  personalization query must never block a scheduled post.

The mode is picked once per post, before the writer prompt is built, and is
not re-picked on the one retry attempt.

## Format degradation ladder

Only `story` (plain text) exists today — there is nothing to degrade to. A
format→fallback mapping (`photo → story`, `video_note → voice → story`) is
introduced as later releases add media formats; a media failure will demote
the post, never kill it.

## Scheduling — `src/jobs/life_post.py`

- Exactly `LIFE_POSTS_PER_WEEK = 2` posts per ISO week, on random days at a
  random minute inside `LIFE_POST_WINDOW` (10:00–22:00, Moscow Time —
  `Europe/Moscow`, fixed UTC+3, no DST). `week_plan(now)` is a deterministic seeded plan
  (`random.Random(f"life-{iso_year}-{iso_week}")`) — no schedule table, and
  calling it at any point during the week returns the same plan.
- `life_post_job` runs daily at the window start (`LIFE_POST_RUN_TIME`,
  10:00 local) and schedules a one-off `run_once` at the planned minute when
  today is one of the week's two planned days.
- `catch_up_life_post_job` runs once at startup (+60s):
  - no episode has ever been posted → this is a fresh deployment; the very
    first life post is scheduled immediately (or deferred to the next
    daytime window if the bot started at night) — Жора's opener.
  - otherwise, recovers a missed scheduled slot the same way, comparing
    `bot_memories.get_latest_posted_at()` against `most_recent_due_slot`.
- Night is quiet hours for **proactive** posts only — the reactive pipeline
  (mentions, replies) is untouched and answers around the clock.

## Canon write rules

Canon (`bot_memories`) is written only by this scheduled job — chat members
cannot inject it. The episode writer sees the last 10 episodes plus a
20-newest/10-sampled-older slice of facts (never the whole store), so canon
holds months of detail without blowing the prompt budget. `current_activity`
is stored on the episode row and is the sole source for the pipeline's
«что делаешь сейчас» answer (see `src/pipeline/README.md`).

## Shared helper

`load_json_object` (used to parse the writer's JSON output) lives in
`src/utils/llm_json.py`, extracted from `src/jobs/roles.py` so both jobs
share one implementation.
