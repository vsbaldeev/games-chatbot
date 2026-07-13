Жора's scheduled life posts — the character's own story, told to every chat twice a
week and remembered as canon — plus a silent daily activity refresh that keeps
"what are you doing right now" fresh between posts.

## Flow

```
src/jobs/life_post.py (schedule) ──► src/life/poster.py:post_life_episode(bot)
    │
    ├─ src/life/writer.py: EpisodeWriterAgent.write_episode()
    │      reads bot_memories.get_recent_episodes(10) + get_writer_facts()
    │      (20 newest + 10 sampled older) + get_recent_activities(7);
    │      src/life/engagement.choose_mode() picks how this post engages
    │      the chat (see below) → prompts EPISODE_WRITER_SYSTEM (today's
    │      date/season + dated recent activities via calendar_ru, for
    │      season-appropriate and non-contradicting episodes) → strict JSON
    │      {episode_text, image_prompt, voice_script, voice_teaser,
    │        current_activity, format} → parse_episode validates lengths
    │        (episode_text ≤ 450, voice_script ≤ 500, voice_teaser ≤ 120)
    │        and required fields; one retry on a malformed/invalid
    │        response, then None (slot skipped, catch-up retries later)
    │
    ├─ resolve_voice_media: for a voice episode, synthesize the spoken story
    │      once (prepare_tts_text + speech_service.synthesize); any failure
    │      (service not ready, unspeakable script, synthesis error) demotes
    │      the episode to a story — a media failure never kills a post
    │
    ├─ send_episode: fan out to every chat (achievements.get_all_chat_ids,
    │      asyncio.gather(..., return_exceptions=True) — one chat's failure
    │      cannot abort the others), record each send in unified_messages
    │      (content = episode_text even for voice posts, so the bot's own
    │      posts need no transcription/vision when a member replies to them)
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

## Voice posts — teaser caption, story in the voice

People skip long voice notes that give no reason to press play, so a voice
post never shows its full text. The caption is only `voice_teaser` — one dry
hook line in Жора's style («Про медведя, мёд и одну плохую идею.»), never a
summary — while the full episode lives in the spoken `voice_script`, which
must be self-contained and carry the engagement question/mention (in MEMBER
mode the teaser may hint «тут кое-что про одного из вас» but only the voice
names the member). Scripts are capped at `EPISODE_VOICE_SCRIPT_MAX_CHARS`
(500 ≈ 25–40 s of Silero speech, tighter than the general `TTS_MAX_CHARS`
synthesis limit) so the duration Telegram shows before playing stays a
low-commitment tap. Reply-context needs no transcription:
`unified_messages.content` records the full `episode_text` even though
members see only the teaser.

## Format degradation ladder

`voice → story`: the voice payload is built once before the fan-out
(`resolve_voice_media`), and any media failure demotes the episode to a text
story — the recorded format is the demoted one, so the never-repeat-format
rule sees what was actually posted. Later releases extend the mapping
(`photo → story`, `video_note → voice → story`); a media failure always
demotes the post, never kills it.

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

Canon (`bot_memories`) is written only by scheduled jobs — chat members
cannot inject it. The episode writer sees the last 10 episodes plus a
20-newest/10-sampled-older slice of facts (never the whole store), so canon
holds months of detail without blowing the prompt budget. `current_activity`
is stored on both `episode` rows (life posts) and `activity` rows (the daily
refresh below); whichever is newest is the pipeline's answer for «что делаешь
сейчас» (see `src/pipeline/README.md`).

## Daily activity refresh — `src/life/activity.py`, `src/jobs/daily_activity.py`

Life posts land only twice a week, so between them `current_activity` used
to sit frozen for days (the exact "рубит дрова for a week" bug this refresh
fixes) and then go stale and get improvised inconsistently. A lightweight
daily job closes that gap without ever posting to chat:

```
src/jobs/daily_activity.py (09:30 MSK, before the 10:00 life-post window)
    │ skip if bot_memories.get_current_activity() already dates to today
    │   (an earlier refresh, or a life post that already landed)
    ▼
src/life/activity.py: refresh_daily_activity()
    reads bot_memories.get_recent_activities(10) + get_facts(5)
    → build_activity_prompt: today's date/weekday/season (calendar_ru) +
      dated recent activities ("не повторяй их") + canon facts
    → DAILY_ACTIVITY_SYSTEM (ChatGroq, config.ACTIVITY_MODEL, temp 0.9,
      single call, up to 2 attempts) → one bare phrase, not JSON
    → parse_activity_phrase: strips think-blocks/quotes, drops (retry) if
      empty or over CURRENT_ACTIVITY_MAX_CHARS (80)
    → bot_memories.insert_activity(phrase) — kind='activity', capped at
      MAX_BOT_ACTIVITIES (30, oldest pruned)
```

Fails soft: any exception or two unusable attempts just leave yesterday's
activity in place (it ages from "fresh" into "recent" phrasing) — a broken
refresh must never crash the job. `catch_up_daily_activity_job` recovers a
refresh missed while the bot was down, same as the life-post catch-up.

`src/life/calendar_ru.py` renders the shared Russian date/weekday/season and
relative-day ("вчера", "позавчера", "8 июля") phrasing consumed by this
generator, the episode writer (season-appropriate episodes, continuity with
recent activities) and the response prompt (dated "what did you do
yesterday" answers via `bot_recent_activities`, see `src/pipeline/README.md`).

## Shared helper

`load_json_object` (used to parse the writer's JSON output) lives in
`src/utils/llm_json.py`, extracted from `src/jobs/roles.py` so both jobs
share one implementation.
