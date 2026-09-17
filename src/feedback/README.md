# Feedback

Implicit user-feedback tracking on bot messages: reactions, replies, and
correction detection — the online proxy for how often the bot says
something wrong, and how much members engage with it. Design:
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
                 depth. A depth-1 (direct) reply that is text is additionally
                 classified for a correction — the reply's own row is already
                 in the fetched chain, so no second query is needed.

classifier.py — classify_correction(bot_text, reply_text): one TAG_MODEL
                 call (reasoning_effort="none", max_tokens=10) returning
                 True only on an explicit CORRECTION verdict; fails closed
                 (False) on any error or unparseable output.
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

## Known limits

- Correction rate is a lower bound: it misses corrections sent without
  Telegram's reply feature, non-text replies, and anything the classifier
  gets wrong.
- Animation (GIF) replies and duel-game replies don't pass through the
  router, so they never reach `track_reply`.
