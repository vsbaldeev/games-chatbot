# Feedback

Implicit user-feedback tracking on bot messages: reactions and replies, kept as
raw counters on one row per bot message — engagement metrics and the online
proxy for the bot's impact on the chat. No rates are stored; every figure is a
`SUM` taken at query time.

Lives outside `src/events` because `src/pipeline` (the call site for
`replies.py`) never imports `src/events` — the dependency runs the other way.

## How it works

One table, `bot_message_feedback` (schema in `src/store/README.md`), one row per
bot message. The row is created when the message is sent
(`src.store.message_feedback.register`) and carries seven counters:

```
positive_reactions   emoji in ❤👍😂🔥💯😅😁 added (−1 when removed)
negative_reactions   any other emoji added (−1 when removed)
positive_replies     text reply the labeller called POSITIVE
negative_replies     text reply the labeller called NEGATIVE
neutral_replies      text reply the labeller called NEUTRAL
media_replies        reply with no text — sticker, GIF, photo, voice, video note
unlabelled_replies   UNCLEAR label, blank text, unparseable answer, or failed call
```

`ignored` is TRUE exactly when every counter is 0. Every mutating statement
recomputes it, so a message that is reacted to and then un-reacted goes back to
ignored. There is no time window: counters move whenever feedback arrives, for as
long as the row exists (60-day retention).

## Modules

```
replies.py    — track_reply(chat_id, message_id, bot_id): called from
                 src.pipeline.router on every reply, for every media type.
                 Walks the reply chain (src.store.unified_messages.get_chain),
                 finds the nearest bot message, and spawns a background task
                 (count_reply) so no model call ever sits on the reply path.
                 The task skips replies to EXCLUDED_TRIGGERS ("social_link"),
                 classifies the reply (classify_reply: non-text → media, blank
                 → unlabelled, otherwise one label_reply call) and bumps one
                 counter via message_feedback.add_reply. Only the nearest bot
                 message is credited. Never raises.

reply_labeller.py — build_payload/parse_label/label_reply: one OpenRouter call
                 labels one text reply POSITIVE/NEGATIVE/NEUTRAL/UNCLEAR.
                 OpenRouter-only on purpose: the labeller must never share the
                 Groq quota that serves chat replies. Parsing is strict; an
                 answer outside the contract returns None (counted as
                 unlabelled). A call that never reached a model raises
                 LabellingCallFailed (also counted as unlabelled, and logged).

sentiment.py  — POSITIVE_REACTIONS (❤👍😂🔥💯😅😁) and
                 is_positive_reaction/normalize_reaction. Every other
                 reaction counts as negative — the operator's rule for this
                 chat, not an oversight. Normalizes U+FE0F so "❤️" and "❤"
                 compare equal.
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
- Retention: `src/jobs/cleanup.py`.

## Explicit limits

- Duel-game replies don't pass through the router, so they never reach
  track_reply. (Animation replies do, since handle_animation_message routes
  through run_pipeline.)
- Negative replies are a lower bound: dismissals sent without Telegram's reply
  feature are invisible, which is likely the largest blind spot.
- `media_replies` is not a sentiment. It counts wordless replies and asserts
  nothing about what the sender meant.
- Changing `POSITIVE_REACTIONS` only affects reactions counted afterwards: emoji
  are not stored individually, so past rows cannot be recounted.
- A deleted reply is never uncounted; only reactions can be removed.
- `social_link` messages (Reels and long-form YouTube summaries) have rows but
  never accumulate reply counters; their reactions are still counted.
- The newest messages read as ignored until someone responds.
- A failed or unclear label is counted in `unlabelled_replies`, not retried. Read
  the three sentiment-reply counters together with it.

## Reading the numbers

Counts only — divide yourself when you want a rate. Read them split by
`trigger`: `explicit` means a member addressed the bot and it answered, and is
the quality signal; `youtube_short` means the bot spoke unprompted because
someone pasted a link, so its ignored count is structurally high and says
nothing about whether the bot is liked.

```sql
SELECT trigger,
       COUNT(*)                        AS messages,
       COUNT(*) FILTER (WHERE ignored) AS ignored,
       SUM(positive_reactions)         AS positive_reactions,
       SUM(negative_reactions)         AS negative_reactions,
       SUM(positive_replies)           AS positive_replies,
       SUM(negative_replies)           AS negative_replies,
       SUM(neutral_replies)            AS neutral_replies,
       SUM(media_replies)              AS media_replies,
       SUM(unlabelled_replies)         AS unlabelled_replies
FROM bot_message_feedback
WHERE sent_at >= now() - interval '7 days'
GROUP BY trigger
ORDER BY messages DESC;
```

Add `AND source = 'pipeline'` to see only conversational messages, excluding
memes, selfies, weekly roles and notices. Those carry `trigger = '-'` and land in
their own row otherwise.
