# Feedback

Implicit user-feedback tracking on bot messages: reactions, replies, and
per-reply reaction labels — engagement metrics and the online proxy for the
bot's impact on the chat. Design:
`docs/superpowers/specs/2026-09-17-bot-feedback-metrics-design.md`.

Lives outside `src/events` because `src/pipeline` (the call site for
`replies.py`) never imports `src/events` — the dependency runs the other way.

## Modules

```
replies.py    — track_reply(chat_id, message_id, bot_id): called from
                 src.pipeline.router on every reply, for every media type.
                 Walks the reply chain (src.store.unified_messages.get_chain)
                 and credits every bot ancestor via
                 src.store.message_feedback.add_reply, each at its own hop
                 depth. Text replies are enqueued for offline labelling; the
                 job drains the queue asynchronously.

sentiment.py  — POSITIVE_REACTIONS (❤👍😂🔥💯😅😁) and
                 is_positive_reaction/normalize_reaction. Every other
                 reaction counts as negative — the operator's rule for
                 this chat, not an oversight. Normalizes U+FE0F so "❤️"
                 and "❤" compare equal; the rollup SQL mirrors that
                 normalization.

reply_labeller.py  — build_batch_payload/parse_label_batch/label_batch: one
                     OpenRouter call labels up to REPLY_LABEL_BATCH_SIZE text
                     replies POSITIVE/NEGATIVE/NEUTRAL/UNCLEAR, plus a cause
                     on the negatives that is stored and never reported.
                     MEDIA is not in the model's vocabulary — the job applies
                     it by rule to every wordless reply. OpenRouter-only on
                     purpose: an offline batched job must never share the Groq
                     quota that serves chat replies. Parsing is strict; a
                     malformed item leaves its row pending rather than storing
                     a guess.
```

## Where the rest lives

- Reaction tracking: `src/events/reactions.py` (`handle_message_reaction`,
  a `MessageReactionHandler` — group-only, requires the bot to be a group
  admin).
- Registration of every bot message for tracking: `src.store.message_feedback.register`,
  called from `send_and_store`/`edit_and_store` (`src/events/sending.py`),
  `deliver_and_record` (`src/events/messages.py`), `src/memes/sender.py`,
  `src/life/selfie.py`, and `src/jobs/roles.py`.
- The response LLM trace (exact prompt/response per pipeline reply):
  `src.store.llm_log`/`llm_system_prompts`, written by `deliver_and_record`.
- The 15-minute window close and hourly rollup: `src/jobs/feedback.py`.
- Retention: `src/jobs/cleanup.py`.

## Explicit limits

- Duel-game replies don't pass through the router, so they never reach
  track_reply. (Animation replies do, since handle_animation_message routes
  through run_pipeline — the earlier claim that they didn't was stale.)
- negative_response_rate is a lower bound: it misses dismissals sent without
  Telegram's reply feature, which is likely the largest blind spot.
- media_response_rate is not a sentiment. It counts wordless replies — GIF,
  sticker, voice, video note, photo. In this chat they lean negative, but the
  metric asserts nothing, and folding that inference into
  negative_response_rate was rejected on purpose.
- The seven rates say that members reacted, not why. The cause is stored on
  bot_reply_labels and aggregated by nothing; querying it by hand is the only
  route to attribution, and it has no validated accuracy.
- Correction tracking is gone, and this is the one irreversible part of the
  change. v5_0004 did not stop writing `corrected` and `correction_rate` — it
  DROP COLUMNed them. They do not exist, they are not NULL, and the roughly
  three and a half weeks of history they held (bot_feedback_metrics only
  started filling on 2026-09-17) was destroyed, not preserved. Migrations here
  are forward-only, so there is no route back to it. The same migration dropped
  six more metric columns and four per-message counters the same way — see
  src/store/README.md.
- FACTUAL as a cause records that a member thinks the bot got something
  wrong, not a verified check. Factual groundedness is deliberately not
  measured.

## Checking the labels

Two automatic checks, both requiring no hand-labelling, plus the queries the
operator actually reads week to week.

### Two-model agreement

`scripts/compare_label_models.py` labels the same pending replies with each
`REPLY_LABEL_MODELS` leg independently and reports where they disagree. High agreement does not prove correctness — two models
can be wrong identically — but disagreement reliably locates the ambiguous
cases, which is what a prompt revision needs to see. Read-only: it writes
nothing to `bot_reply_labels`. Run it whenever `REPLY_LABEL_SYSTEM` or the
model list changes:

```
python -m scripts.compare_label_models --rows 40
```

### Emoji cross-check

Where a bot message drew both a model-labelled text reply and at least one
reaction, do the two signals agree? Both sides are now
binary, so this is directly comparable: POSITIVE labels sitting under
negative-only reactions is the drift signal. Free ground truth — nobody
labelled the emoji. This reads `bot_message_feedback.emojis` directly rather
than the rollup, because the rollup aggregates per hour and this comparison
needs per-message pairing. Bind `$1` to the same list
`src.feedback.sentiment.positive_reactions_sql_array()` returns, so nobody
retypes the emoji into the query:

```sql
SELECT
    label.reaction,
    COUNT(*) FILTER (WHERE EXISTS (
        SELECT 1 FROM jsonb_object_keys(feedback.emojis) AS emoji_key
        WHERE replace(emoji_key, U&'\FE0F', '') = ANY($1::text[])
    )) AS with_positive_emoji,
    COUNT(*) FILTER (WHERE EXISTS (
        SELECT 1 FROM jsonb_object_keys(feedback.emojis) AS emoji_key
        WHERE replace(emoji_key, U&'\FE0F', '') <> ALL($1::text[])
    )) AS with_negative_emoji
FROM bot_reply_labels AS label
JOIN bot_message_feedback AS feedback
  ON feedback.chat_id = label.chat_id
 AND feedback.message_id = label.bot_message_id
WHERE label.labelled_by = 'model'
  AND feedback.emojis <> '{}'::jsonb
GROUP BY label.reaction
ORDER BY label.reaction;
```

### Weekly reporting

The weekly rejection rate and the cause breakdown — the headline queries the
operator actually reads:

```sql
-- The weekly reading, split by trigger. Counts are summed and divided once,
-- never averaged across hourly rates.
--
-- GROUP BY trigger is not optional. "explicit" means a member addressed the
-- bot and it answered; "youtube_short" means the bot spoke unprompted because
-- someone pasted a link. Those have completely different baselines — nobody
-- replying to a summary they never asked for is normal, not rejection — so
-- pooling them averages the one signal that matters (negative_response_rate
-- on "explicit") into the one that does not.
--
-- "social_link" never appears: ROLLUP_SQL already excludes it, and the
-- labelling job marks its replies SKIPPED. The clause below is belt and
-- braces, kept because an explicit filter documents the exclusion at the
-- point someone reads the numbers.
SELECT
    trigger,
    SUM(messages)                                 AS messages,
    SUM(ignored)::real / SUM(messages)            AS ignore_rate,
    SUM(media_responded)::real / SUM(messages)    AS media_response_rate,
    SUM(positive_emoji)::real / SUM(messages)     AS positive_emoji_rate,
    SUM(negative_emoji)::real / SUM(messages)     AS negative_emoji_rate,
    SUM(positive_responded)::real / SUM(messages) AS positive_response_rate,
    SUM(negative_responded)::real / SUM(messages) AS negative_response_rate,
    SUM(neutral_responded)::real / SUM(messages)  AS neutral_response_rate,
    SUM(unlabelled)::real / SUM(messages)         AS unlabelled_rate
FROM bot_feedback_metrics
WHERE bucket_start >= now() - interval '7 days'
  AND trigger <> 'social_link'
GROUP BY trigger
ORDER BY messages DESC;
```

Add `AND source = 'pipeline'` when you want only conversational messages,
excluding memes, selfies, weekly roles and notices. Those carry
`trigger = '-'` and land in their own row otherwise.

Notes:

- The response rates are only as trustworthy as `unlabelled_rate` is small.
  Read them together or not at all.
- `trigger` splits solicited from unsolicited messages. `explicit` is the
  quality signal; `youtube_short` is a summary nobody asked for, so its
  `ignore_rate` is structurally high and means nothing about whether the bot
  is liked. `social_link` is not measured at all and never appears.
- `media_response_rate` counts wordless replies and asserts nothing about
  what the sender meant.
- To check the labeller is not dropping replies, count `bot_reply_labels`
  directly rather than reaching for a rollup column. The per-reply grain is
  where that question belongs:

```sql
-- Every text reply should carry exactly one of POSITIVE, NEGATIVE, NEUTRAL,
-- UNCLEAR or SKIPPED. A non-zero NULL row means rows are stuck pending; that
-- is the bug. A large SKIPPED count is expected and healthy — those are
-- replies to excluded triggers that cost no model call.
SELECT reaction, COUNT(*) AS replies
FROM bot_reply_labels
WHERE media_type = 'text'
  AND created_at >= now() - interval '7 days'
GROUP BY reaction
ORDER BY replies DESC;
```

One more query, documented as the escape hatch rather than a metric —
nothing aggregates `cause`, so this is the only way to see it:

```sql
-- Not a metric. Unvalidated attribution, for when negative_response_rate is
-- high and you need a hypothesis about why. Treat the ordering as a hint.
SELECT label.cause, COUNT(*) AS replies
FROM bot_reply_labels AS label
WHERE label.reaction = 'NEGATIVE'
  AND label.labelled_by = 'model'
  AND label.created_at >= now() - interval '30 days'
GROUP BY label.cause
ORDER BY replies DESC;
```

This breakdown has no human-validated accuracy figure, by design — a
stratified hand-check of ~40 rows is the cheapest way to put error bars on
it, if it is ever about to drive a product decision.
