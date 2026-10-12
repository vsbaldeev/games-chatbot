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
- Correction tracking is gone. corrected and correction_rate hold history and
  are NULL for every bucket after v5_0004.
- FACTUAL as a cause records that a member thinks the bot got something
  wrong, not a verified check. Factual groundedness is deliberately not
  measured.
