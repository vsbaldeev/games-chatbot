LangGraph StateGraph pipeline — processes every incoming Telegram message through typed
nodes. Each node reads BotState and returns a partial update dict.

---

## Overview

```
router ──► ingester ──► filter ──► guard ──► context_builder ──► worker ──► response ─┬─► memory_writer ──► END
                                                                                        └─► language_correction ──► memory_writer ──► END
   └─► humor ──► memory_writer ──► END   (autonomous joke when the humor gate fires)
```

Conditional exits exist at every node — see the detailed diagrams below.

---

## Router

Stores every message and decides whether the bot should respond.

```
incoming message
    │
    ├─ always: insert into unified_messages
    │              text   → content = raw_text
    │              photo  → content = "[photo]" or "[photo]\n<caption>" (placeholder marks
    │                       row as still needing vision description; caption preserved)
    │              voice / video_note / video → content = placeholder, file_id stored
    │
    ├─ text
    │     ├─ YouTube Shorts link      → should_respond=True,  trigger="youtube_short"
    │     │       (checked first, even for forwarded messages — forwarding is
    │     │        the dominant way links arrive, and summarizing a video does
    │     │        not put words in the sender's mouth; gated by shorts.py:
    │     │        same video id in the same chat once per 24h (dedup_gate),
    │     │        SHORTS_DAILY_CAP=15 summaries per chat per sliding 24h
    │     │        window — a gated link falls through to the rules below,
    │     │        costing zero downloads and zero LLM tokens)
    │     ├─ @bot_username in text    → should_respond=True,  trigger="explicit"
    │     │       (word-boundary regex via is_explicitly_addressed — URLs and
    │     │        longer words containing the username do not count)
    │     ├─ reply to bot message     → should_respond=True,  trigger="explicit"
    │     ├─ word «бот»/"bot" in text → should_respond=True,  trigger="insult_check"
    │     │       (BOT_WORD_RE, cheap regex precondition; the filter node then
    │     │        decides whether the message actually insults the bot)
    │     └─ otherwise               → should_respond=False, trigger="random"
    │
    ├─ voice / video_note / video / photo
    │     ├─ @bot_username in caption → should_respond=True,  trigger="explicit"
    │     ├─ reply to bot message     → should_respond=True,  trigger="explicit"
    │     ├─ album item whose media_group_id already rolled (5-min TtlGate)
    │     │                           → should_respond=False (one roll per album)
    │     └─ otherwise               → random.random() < MEDIA_RESPONSE_CHANCE (0.10)
    │
    └─ sticker / animation / audio   → should_respond=False (stored as placeholder)
    │
    ├─ every text message → humor_gate.observe(chat_id)   [counts toward joke cadence]
    │
    ├─ should_respond=False + non-forwarded text (≥ 20 chars)
    │       → asyncio.create_task(extract_and_save)   [passive memory, background]
    │         forwarded messages are skipped — channel content must not be
    │         attributed as facts about the person who forwarded it
    │
    ├─ should_respond=False + humor gate fires → humor   [autonomous joke]
    ├─ should_respond=False → memory_writer (long text) or END
    └─ should_respond=True  → ingester
```

### Autonomous humor gate

`humor_gate` (no LLM) decides whether an un-addressed message is worth handing to
the comedian, keeping the model off the per-message hot path. It fires only when
all hold: joke-worthy plain text, ≥ `MIN_MESSAGES_SINCE_JOKE` messages since the
last joke, the `COOLDOWN_SECONDS` window elapsed since the last *sent* joke, and a
low probability roll — tuned for "rare & sharp" so the bot opens lulls and never
spams.

When it fires, the `HumorNode` (`humor_node.py`) renders the recent conversation
with `[#message_id]` markers (forwarded rows carry a `[переслал]` marker so the
comedian can tell shared channel content from a participant's own words), gathers
participant material (`src/agent/roast_material.py`), and asks the `ComedianAgent`
(`src/agent/comedian.py`) for a strict-JSON decision that includes a `reply_to`
citation — the id of the message the joke is actually about. On `act` the node
validates the citation against the fetched messages (a hallucinated id or a
citation of the bot's own message degrades to no anchor), drops the joke
entirely when the cited target's author is wound down by the engagement gate
(`engagement_gate.is_wound_down`, read-only score peek — bot-initiated humor
must not restart a conversation the gate is ending), sets
`state["response"]`, `response_trigger="humor"` and `humor_reply_to_msg_id`, and
stamps the cooldown via `mark_joke_sent`. `run_pipeline` then anchors the joke as
a Telegram reply to the cited message, or posts it un-anchored when there is no
valid target — never as a reply to whichever message happened to trigger the
gate. On an abstain or any error the node stays silent and calls
`mark_considered`. Either way the graph continues to `memory_writer`. The
comedian defaults to silence and only acts when it has a conversation-spawning
hook (light by default, roast sparingly). Jokes are grounded in the rendered
conversation itself — what people actually wrote, contradictions and irony
between messages (including between what someone forwarded and what they then
said) — with retro/nostalgia references allowed only as rare flavor when they
genuinely fit the topic.

---

## Media processing

Two places process media into text. The ingester handles the current message;
the context builder lazily processes media found in reply chains.

```
ingester (current message, should_respond=True only)
    ├─ text       → processed_text = raw_text
    │     trigger="youtube_short": the short is downloaded via yt-dlp
    │     (in-process Python API, run_in_executor; muxed 360p mp4, no ffmpeg;
    │     limits: ≤180s, ≤25MB (Groq Whisper cap), 90s total timeout) and fed
    │     through the same Whisper + frame pipeline as Telegram videos, plus
    │     the top 10 comments (top-sorted, ≤200 chars each) as audience
    │     reaction; transcript capped at 2000 chars
    │     processed_text = user text + "\n\n[YouTube Shorts «title», канал X,
    │     N сек]\n[Аудио]: …\n[Видео 1/3]: …\n[Топ-комментарии зрителей]: …"
    │     and unified_messages is updated so reply chains show the content;
    │     youtube_short_content is set as the success flag (None on failure —
    │     no transcript AND no frames counts as failure; title/comments alone
    │     are not enough to react honestly)
    │     PO tokens for YouTube bot-detection come automatically from the
    │     pot-provider docker-compose sidecar via the bgutil yt-dlp plugin
    ├─ voice      → Groq Whisper → transcript
    ├─ video_note → Groq Whisper + frame extraction (see below); frames' vision
    │               calls also yield media_is_real_person (majority vote — see below)
    ├─ video      → Groq Whisper + frame extraction (see below); same
    │               media_is_real_person aggregation (unused for gating today —
    │               only photo/video_note are gated, see filter below)
    ├─ photo      → vision LLM description; combined with caption when present
    │               "<description>\n(подпись: <caption>)" form
    │               all non-text results: update unified_messages content
    │               the vision prompt is hedged: names/titles are stated only
    │               when the model is confident; otherwise it describes the
    │               scene without guessing, and the response prompt forbids
    │               asserting unconfirmed names from the description as fact
    │               short visible text (meme captions, реплики, headlines) is
    │               quoted verbatim in its original language, so requests like
    │               «переведи» have the actual text to work with
    │               the same vision call also yields media_is_real_person
    │               (see below) — no extra LLM round trip
    └─ sticker    → vision LLM description, only when should_respond=True
                    (plain sticker traffic is enriched lazily instead — see
                    below; note the router currently never responds to
                    stickers, so in practice all sticker enrichment is lazy)

real-person-vs-meme classification (ingester.parse_vision_response)
    every vision call (photo, and each extracted video/video_note frame) is
    instructed to prepend a [ЧЕЛОВЕК]/[МЕМ] tag before its description
    (VISION_PROMPT); the tag is parsed off and never leaks into the stored
    description. For photo it is used directly; for video/video_note the
    per-frame tags are majority-voted (aggregate_real_person) into one
    media_is_real_person verdict, None when no frame could be classified.
    Surfaced in BotState as media_is_real_person (None for text/voice or on
    any classification failure — fails open). Consumed only by the filter
    node's random-trigger meme gate (see Safety below); explicit @mentions
    and replies ignore it entirely, and lazy reply-chain photo enrichment
    (enrich_photo_row) discards it — only the currently incoming message's
    classification can gate a response.

transcription (Groq Whisper, verbose_json, temperature=0)
    language pinned to Russian (WHISPER_LANGUAGE) — short notes no longer flip
    into a random language; genuinely English notes transcribe degraded
    garbage detection: a transcript is discarded (treated as empty) when every
    segment shows no-speech probability > 0.6, avg_logprob < −1.0, or
    compression_ratio > 2.4, or when a short transcript matches Whisper's
    known silence boilerplate («Продолжение следует…», «Спасибо за просмотр»);
    rejections are info-logged for threshold tuning

frame extraction (PyAV)
    duration < 15s   → 1 keyframe at 50%
    15s – 120s       → 3 keyframes at 25%, 50%, 75%
    > 120s           → audio only, no frames
    output: "[Аудио]: <transcript>\n[Видео 1/N]: <desc>\n[Видео 2/N]: <desc>…"
    without frames the transcript is labelled "[Аудиодорожка видео — возможно
    музыка или речь за кадром]:" so lyrics are not read as the sender's words

shared lazy media enrichment (ingester.enrich_media_row, on demand)
    used by the filter node (replied-to row, before classification) and by
    context_builder (every photo/sticker row in the reply chain); other media
    types pass through untouched:
    photo (enrich_photo_row):
        [photo] / [photo]\n<caption> → describe_photo(file_id) → combined with caption
                                       → update unified_messages (cached for future replies)
        Detection: content.startswith("[photo]"); after enrichment content begins with the
        description so re-enrichment is skipped automatically — whichever node
        enriches first, the other reuses the cached content.
    sticker (enrich_sticker_row):
        [sticker] → describe_sticker(file_id) → update unified_messages
        Descriptions are additionally cached in the sticker_descriptions table
        keyed by Telegram's file_unique_id (stable across resends and bots) —
        a resent sticker costs one get_file call and zero vision calls.
        By payload kind (magic bytes; .tgs short-circuited via file_path):
          static WEBP/PNG → vision LLM directly
          video WEBM      → PyAV keyframe extraction + frame description
          animated .tgs   → left as [sticker] (Lottie JSON, not renderable)
    Rows without file_id (e.g. update fallbacks, old records) are left as-is.
```

---

## Safety (filter → guard)

```
filter  (runs after ingester)
    ├─ trigger="youtube_short" → deterministic bypass, no LLM classification
    │       (a bare link would classify MEANINGLESS and be dropped)
    │       ├─ youtube_short_content set   → pass through to guard
    │       ├─ empty + sender explicitly addressed the bot
    │       │       → canned «Не смог посмотреть…» (SHORTS_FAILED_REPLIES)
    │       └─ empty + unaddressed → should_respond=False, full silence
    │               (no emoji reaction — the bot was never addressed)
    ├─ media message, processed_text empty
    │       ├─ explicit trigger → honest canned reply, no LLM
    │       │     voice/video → «Не расслышал…» (TRANSCRIPTION_FAILED_REPLIES)
    │       │     photo       → «Не разглядел…» (VISION_FAILED_REPLIES)
    │       └─ random trigger → should_respond=False + random emoji reaction
    ├─ media message, processed_text non-empty
    │       ├─ random trigger, media_type in (photo, video_note), and
    │       │     media_is_real_person is False (a meme, not a real person —
    │       │     see ingester's real-person-vs-meme classification above)
    │       │       → should_respond=False, fully silent — as if the random
    │       │         roll had simply missed (is_meme_random_trigger)
    │       └─ otherwise → pass through (explicit @mentions/replies are
    │             never gated — a member directly asking the bot to react
    │             to a meme still gets the roast)
    ├─ text, raw_text empty  → should_respond=False (silent)
    addressed messages (trigger="explicit", FILTER_SYSTEM prompt):
    │   if the message is a reply, the replied-to message is loaded from
    │   unified_messages and shown to the classifier as context (fails soft
    │   to no-context), so short reactions to the bot's own messages —
    │   «ахаха что?», «поясни» — classify as MEANINGFUL, not MEANINGLESS
    │   a replied-to photo or sticker still in placeholder form (e.g. the
    │   bot's own posted meme) is vision-enriched first
    │   (ingester.enrich_media_row, cached), so «переведи» under a meme is
    │   judged against the image's actual content; when enrichment is
    │   impossible (no file_id, vision failure, animated sticker) or the row
    │   is another bare media placeholder ([voice], [animation]…), the
    │   placeholder is hidden and the reply classifies context-free instead
    │   of against a token the classifier cannot see
    ├─ text, LLM → MEANINGLESS or BANTER
    │       ├─ text looks like a question or request («?», more than
    │       │   SUBSTANTIVE_WORD_COUNT non-laughter words, leading
    │       │   interrogative, or imperative request verb like «переведи»/
    │       │   «расскажи»/«поищи»/«загугли»; laughter tokens skipped) →
    │       │   overridden to MEANINGFUL: every MEANINGLESS/BANTER category
    │       │   is a SHORT reaction, so a longer message is never meaningless
    │       └─ otherwise → engagement gate (see wind-down engine below)
    ├─ text, LLM → BOT_INSULT (insult/provocation aimed at the bot) → engagement gate
    ├─ text, LLM → MEANINGFUL → engagement gate
    └─ text, LLM error       → should_respond=True (fails open)

    overheard messages (trigger="insult_check", OVERHEARD_SYSTEM prompt —
    text mentioned the word «бот»/"bot" without addressing the bot):
    ├─ classifier input includes the last 5 chat messages (fails soft to bare
    │    text) so «бот» resolves to the right referent — game bots, other
    │    Telegram bots and people playing «как бот» classify as OTHER
    ├─ LLM → BOT_INSULT → confirmed by the stronger INSULT_CONFIRM_MODEL
    │    (llama-3.3-70b-versatile) on the same input; only agreement acts —
    │    disagreement or a confirmation error resolves to silence
    │       → engagement gate (as BOT_INSULT)
    └─ anything else / LLM error → should_respond=False, silent drop
            (no emoji — the bot was never addressed; long texts still get
             passive memory extraction, mirroring the router's behaviour)

engagement gate — conversation wind-down engine (engagement_gate.py +
store/engagement.py, table engagement_scores): one leaky-bucket attention
score per (chat_id, user_id), persisted in Postgres so a redeploy never resets
a wound-down user. Every addressed verdict (and every double-confirmed
overheard insult, and explicitly addressed transcribed media as MEANINGFUL)
charges a weight — BOT_INSULT 3.0, BANTER/MEANINGLESS 2.0, MEANINGFUL 1.0 —
decayed with a 30-min half-life in a single atomic UPSERT; the post-charge
score maps onto a tier (brush-off >7, emoji >13, silence >19), so any
sustained conversation fades out like a person losing interest. The
«Оскорблял бота N раз» counter fact in user_memories is still incremented for
every addressed or double-confirmed insult at any tier, via
asyncio.create_task; it feeds weekly roles and roasts only — the context
builder filters counter tallies out of reply prompts so the bot does not
keep reciting the score. Store errors fail open to the full tier.
    ├─ FULL tier      → should_respond=True — full reply; BOT_INSULT sets
    │       is_bot_insult=True (comeback hint, worker skipped; two rapid
    │       insults both land here by design — the classifier has false
    │       positives). A BOT_INSULT that *replies to the bot's own message*
    │       and any BANTER verdict additionally set wind_down=True: a
    │       mirrored counter-insult mid-thread never earns a fresh full
    │       comeback (that is what fuels roast-battle loops)
    ├─ BRUSH_OFF tier → should_respond=True + wind_down=True — the response
    │       node injects a close-the-conversation hint (one short in-character
    │       phrase, no questions, no invitations) and the worker is skipped;
    │       BANTER at this tier degrades straight to the bored emoji reaction
    ├─ EMOJI tier     → should_respond=False + bored emoji reaction
    │       (DISMISSIVE_REACTIONS pool: 🥱 😴 🗿 🤨; MEANINGLESS keeps the
    │       friendly REACTION_POOL until this tier)
    └─ SILENCE tier   → should_respond=False, nothing at all
    │
    ├─ should_respond=False → END
    └─ should_respond=True  → guard

guard   llama-prompt-guard-2-86m
    output contract (verified 2026-07-04 against the live API): the model
    returns a numeric probability-of-malicious string («0.9996» for an
    injection, «0.0004» for a benign greeting) — NOT MALICIOUS/BENIGN labels;
    the score is parsed and blocked at GUARD_SCORE_THRESHOLD = 0.9, and the
    raw label is info-logged per classification for tuning
    classifies typed text only: raw_text (message text or media caption);
    transcripts and vision descriptions are never scanned — they are our own
    models' output, and a voice note cannot inject a prompt
    ├─ text empty / score < 0.9 → blocked=False → context_builder
    ├─ score ≥ 0.9 + trigger="explicit"
    │       → blocked=True, response = random neutral deflection (no guilt
    │         presumed — false positives are inherent at 86M params)
    │       → hack-attempt memory fact only on the 2nd flag per (chat, user)
    │         within 24h (TtlGate hit counter); first flag blocks only  → END
    ├─ score ≥ 0.9 + trigger="random" / "insult_check"
    │       → blocked=True, response=None (silent drop)  → END
    └─ API error / unparsable label → blocked=False (fails open) → context_builder
```

---

## Response pipeline (context_builder → worker → response → memory_writer)

```
context_builder
    ├─ get_recent(limit=20), excluding the current message — always loaded
    ├─ find replied_to message (recent window → get_by_id → replied_to_fallback,
    │    a row-shaped copy of msg.reply_to_message for rows the store never had)
    ├─ get_chain(reply_to_msg_id) → reply_chain (max 10 hops, oldest-first);
    │    empty chain + fallback → one-element chain from the fallback
    ├─ load user_memories facts for all user_ids visible in recent history
    ├─ load initiating user's facts if not already in recent participants
    ├─ load initiating user's weekly role + reason from user_tags → asking_user_tag
    ├─ resolve @mentions (in the question + replied_to) to members and load their
    │    weekly role + reason from user_tags → mentioned_tags
    ├─ bot canon: embed processed_text once, reuse for both queries — top-5
    │    bot_memories.find_similar_facts + 3 newest get_facts (dedupe, cap 8)
    │    → bot_self_facts; top-2 find_similar_episodes above a similarity
    │    floor → bot_self_episodes (Жора's own life canon; see src/life/README.md)
    │    degrades to [] on any failure (embed or DB error) — never fails the pipeline
    ├─ activity gate: the daily refresh means a fresh activity always
    │    exists, so injecting it unconditionally made the bot narrate his
    │    routine in nearly every reply; both activity lookups now run only
    │    when ACTIVITY_QUESTION_RE matches the incoming text («что делаешь /
    │    чем занят / что делал / как дела…») or, for the current activity
    │    alone, on an ACTIVITY_VOLUNTEER_PROBABILITY (10%) roll so he
    │    occasionally volunteers it; gate closed → (None, []) and the model
    │    improvises per the system prompt
    ├─ bot current activity (gate open): newest current_activity across
    │    episode rows (life posts) and activity rows (silent daily refresh,
    │    see src/life/README.md), bucketed by age — < 14h "fresh", < 48h
    │    "recent", older → None (the bot improvises instead of reading a
    │    stale answer) → bot_current_activity
    └─ bot recent activities (asked only, never volunteered):
         bot_memories.get_recent_activities(7), the same newest-first
         (phrase, posted_at) history spanning episode and activity rows,
         degrades to [] on failure → bot_recent_activities
    │
    ▼
worker   ReAct agent with all 13 tools (IGDB, Steam, PS Store, TMDB, AniList, web);
         chain gpt-oss-120b → qwen3.6-27b → gpt-oss-20b (no 8B floor: at that size
         the worker fabricates from memory instead of calling tools; exhaustion
         raises an honest quota error instead)
    ├─ CONTEXT FIRST: if reply chain already contains the answer, no tools called
    ├─ prompt: reply chain (or recent history for explicit triggers) + current question
    │          random triggers receive only the reply chain — no recent history bleed
    ├─ provenance: invoke_worker returns (output, tools_used) from a mechanical
    │          ToolMessage scan → worker_tools_used in state
    ├─ skipped entirely on insult paths (is_bot_insult), wind-down brush-offs
    │          (wind_down — one short closing phrase needs no tools) and Shorts
    │          summaries (trigger="youtube_short" — the source material is
    │          already in processed_text; tools would only add junk) → empty output
    ├─ SearchNotificationCallback sends "🔍 Ищу…" before web_search
    ├─ DailyLimitError → advance_model(), retry with next fallback
    ├─ ContextLengthError → worker_output="" (response node still runs)
    └─ any other error   → worker_output="" (response node still runs)
    │
    ▼
response   personality LLM (ReAct executor, no tools)
    ├─ thread_id = {chat_id}_{root_message_id} for reply chains
    │    (unknown chains fall back to {chat_id}_{reply_to_msg_id})
    ├─ flat (non-reply) mentions skip thread history entirely and are answered
    │    from recent chat context; the exchange is stored under the prospective
    │    chain root {chat_id}_{trigger_message_id} so a follow-up reply chain
    │    starts pre-seeded
    ├─ prompt: thread_history (last 10 turns, thread-scoped, reply chains only)
    │            + user facts + asker's weekly role & reason (asking_user_tag, if any)
    │            + @mentioned members' weekly roles & reasons (mentioned_tags, if any)
    │            + recent history (last 10; random and youtube_short triggers get only
    │              the newest 3, RANDOM_TRIGGER_CONTEXT_LIMIT) + replied_to + worker
    │              findings + current message
    │          worker findings are framed by provenance: «[Собранные данные
    │            (проверено через инструменты)]» when a tool ran, «[Данные из
    │            контекста разговора (во внешних источниках НЕ проверялись)]»
    │            otherwise; the system prompt allows external-world numbers
    │            (prices, player counts, dates) only from the tool-verified frame
    │          when the current message is media (photo/voice/video), its trigger line is
    │            framed as "@user прислал фото. Ниже — его описание… Отреагируй, не пересказывай"
    │            (build_trigger_line) so the model reacts to the vision/transcript description
    │            instead of retelling it as if it were the user's own words
    │          trigger="youtube_short" inverts that framing: nobody has watched the
    │            video yet, so the model is told to retell it in 1–2 sentences and
    │            summarize the audience reaction from the top comments (1–2 sentences);
    │            no worth-watching verdict, no inventing missing details, and no
    │            checking the video's facts against the model's own stale knowledge
    │            (nothing here is tool-verified — the worker is skipped for Shorts)
    │          the bot's own past messages render as "Ты (бот): …" (via row_speaker,
    │            keyed on user_id == BOT_ID) so the model never @mentions or replies to itself
    │          system prompt (RESPONSE_PROMPT) is prepended internally by the executor
    │          genuinely ambiguous requests: when the missing detail would change
    │            the answer (which game, which platform, about whom), the system
    │            prompt tells the model to ask ONE short in-character clarifying
    │            question instead of guessing; mild vagueness gets an answer with
    │            the assumption stated («если ты про PS5-версию — …»); the
    │            insult/wind-down hints override this — they forbid counter-questions
    │          joke requests: when literal execution is pointless in context (e.g.
    │            «переведи» under a meme whose text is already Russian), the system
    │            prompt tells the model to recognize the bit and play along (mock
    │            «translation» from Russian to Russian, needling the addressee)
    │            instead of reciting the text back; genuinely foreign text still
    │            gets a real translation
    │          when someone asks why they (or an @mentioned member) have a role, the
    │          bot explains it from the stored reason
    │          + Жора's own life canon (build_bot_life_lines): relevant canon facts
    │            and past episodes from bot_self_facts/bot_self_episodes (only when
    │            the current message actually touches that topic — background
    │            colour, never filler), plus a current-activity line
    │            («[Прямо сейчас ты]: …» / «[Недавно ты]: …») from
    │            bot_current_activity so «что делаешь сейчас» answers consistently
    │            with the latest life post or daily refresh instead of being
    │            improvised fresh each time, plus a dated
    │            «[Чем ты занимался в последние дни]» block (build_activity_history_lines)
    │            from bot_recent_activities — skipping the newest entry already
    │            shown above — so «что делал вчера/на выходных» answers consistently
    │            too instead of inventing a different past per questioner;
    │            both lines appear only when the ContextBuilder activity gate
    │            opened (asked, or the rare volunteer roll for the current
    │            activity) — most replies carry neither
    │          when the filter set is_bot_insult=True, a hint is injected before the
    │            trigger line telling the model the message is an attack on it and to
    │            answer with a sharp comeback instead of a neutral reply
    │          when the engagement gate set wind_down=True, a hint is injected telling
    │            the model it is bored of this conversation: answer in one short
    │            in-character phrase, close the exchange, no questions or invitations
    ├─ at DEBUG, dumps the exact LLM input before generation (log_response_input):
    │    thread-history turns as one-line excerpts, the assembled final turn
    │    verbatim — the one place to see why a reply went absurd
    ├─ normalizes homoglyphs first (normalize_homoglyphs): Greek/Latin look-alike
    │    letters spliced into a mostly-Cyrillic word (e.g. Greek μ/ά in "тμάксимс")
    │    are mapped back to Cyrillic deterministically, with no extra LLM call;
    │    standalone Greek symbols (π, 50 μg) and pure-Latin words (React) are left intact
    ├─ saves response_messages to state for LanguageCorrectionNode
    ├─ persists the exchange to thread_history only when no correction is
    │    needed — otherwise language_correction persists the corrected reply,
    │    so history always stores what the chat actually saw
    ├─ DailyLimitError / RateLimitError → propagate to top-level handler
    │
    ├─ needs_russian_correction(response) → language_correction. True when the
    │    reply still contains a hard-foreign script (CJK/Hangul/Thai/Arabic/Hebrew,
    │    foreign anywhere) or residual Greek fused into a Cyrillic word that
    │    normalization could not repair (e.g. a bare μ with no Cyrillic twin)
    └─ otherwise → memory_writer
    │
    ▼ (correction path)
language_correction
    ├─ re-invokes ResponseAgent with original response_messages + correction instruction
    ├─ normalizes homoglyphs in the corrected reply before persisting
    ├─ persists the final (corrected or kept-original) reply to thread_history
    ├─ DailyLimitError / RateLimitError → propagate
    └─ any other error → keep original response (silent fallback)
    │
    ▼
memory_writer
    ├─ is_forwarded=True → skip entirely
    ├─ passive (no response): skip if user_message < 20 chars
    └─ asyncio.create_task() — does NOT block the reply
          → qwen/qwen3.6-27b (reasoning disabled) extracts up to 3 new facts
          → source rules: only the user's own words are evidence — the bot's
            reply is context, never a fact source; voice transcripts are
            framed as spoken words; photo/video descriptions are marked as
            NOT the user's words (source_kind: text | voice | media_description)
          → dedup via cosine similarity (fastembed MiniLM-L12, threshold 0.85)
            duplicate → refresh updated_at; new → insert with embedding
          → cap: 30 facts per user per chat, oldest pruned on overflow
          → 90-day expiry: facts (counters included) untouched for 90 days
            are deleted by the nightly cleanup job
          → facts written in Russian
          → cross-user extraction for any @mentioned users (if stripped message ≥ 20 chars):
            sincerity rule (banter/insults are not facts) + facts stored with
            «по словам @X, …» attribution
```

---

## BotState

```python
IncomingMessage:
    chat_id, user_id, username
    raw_text: str | None          # original text or caption
    processed_text: str | None    # transcript / vision description, set by ingester
    media_type: "text" | "voice" | "video_note" | "video" | "photo"
    message_id, reply_to_msg_id, file_id
    is_forwarded: bool            # True when message.forward_origin is set
    media_group_id: str | None    # Telegram album group id
    replied_to_fallback: dict | None  # row-shaped copy of msg.reply_to_message; read-side only

AssembledContext:
    user_facts: dict[str, list[str]]     # username → extracted fact strings (counter tallies like «Оскорблял бота N раз» are filtered out — bookkeeping for roles/roasts, not reply material)
    recent_history: list[dict]           # flat window (last 20), newest-first
    replied_to: dict | None              # the specific message being replied to (for annotation)
    reply_chain: list[dict]              # full reply chain from root to replied-to, oldest-first
    asking_user_tag: dict | None         # {"tag", "reason"} weekly role of the message sender, if any
    mentioned_tags: dict[str, dict]      # username → {"tag", "reason"} for members @mentioned in the question
    bot_self_facts: list[str]            # Жора's own canon facts relevant to this message
    bot_self_episodes: list[str]         # Жора's own past life-post episodes relevant to this message
    bot_current_activity: tuple[str, str] | None  # (phrase, "fresh"|"recent") from the newest life post or daily refresh
    bot_recent_activities: list[tuple[str, float]]  # (phrase, posted_at) history, newest first, for dated "what did you do" answers

BotState:
    incoming: IncomingMessage
    should_respond: bool
    response_trigger: "explicit" | "insult_check" | "random" | "youtube_short" | "humor"
    blocked: bool
    youtube_short_url: str | None      # canonical Shorts URL, set by router
    youtube_short_content: str | None  # labelled transcript/frames/comments block, set by ingester
    media_is_real_person: bool | None  # vision classification for photo/video_note/video, set by ingester; None = text/voice/unclassified
    context: AssembledContext | None
    thread_id: str | None              # {chat_id}_{root_message_id} for replies, chat_id for flat; scopes LLM history
    is_flat_thread: bool               # True when the message is not a reply; skips thread-history reads
    worker_output: str | None
    worker_tools_used: bool            # True when the worker actually ran a tool (mechanical ToolMessage scan)
    search_notification_msg: Any | None  # Telegram Message used as search indicator
    response: str | None
    response_messages: list | None     # LangChain messages passed from response → language_correction
    context_types: ContextTypes        # Telegram context for sending replies
```
