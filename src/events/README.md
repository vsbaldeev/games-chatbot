Telegram event handlers for non-command updates: member tracking, reactions, and messages.

All handlers are registered by EventHandlerManager and MessageHandlerManager in src/bot/handlers.py.

## Handlers

```
members.py
    track_member(update)           — upsert user into chat_members on every update
    handle_new_chat_members(...)   — send welcome message when new user joins
    handle_bot_added_to_chat(...)  — send greeting when bot is added to a new group

reactions.py
    handle_reaction(update)        — map Telegram reaction emoji to user_stats columns
                                     (increment_stat only; counters feed roast material)

messages.py
    handle_message(update)         — text: track stats (night, link, forward, emoji, long),
                                     then enter the LangGraph pipeline
    handle_voice_message(update)   — increment voice_messages + update voice_max_duration,
                                     enter pipeline
    handle_photo_message(update)   — increment photo_messages, enter pipeline
    handle_sticker_message(update) — increment sticker_messages, enter pipeline
    handle_video_message(update)   — increment video_messages, enter pipeline
    handle_animation_message(update) — increment animation_messages (no pipeline)

voice_reply.py
    try_send_voice_reply(msg, text) — answer a voice/video_note trigger in kind:
                                      synthesize the reply via src/tts (Silero v5)
                                      and send it with reply_voice; returns None on
                                      any failure so deliver_response falls back to
                                      the plain text reply
```

## Chat-requested selfies

When the pipeline accepted a photo request (`photo_request` state flag set by
the filter, no selfie already in flight), `run_pipeline` launches
`src/life/selfie.deliver_selfie` as a fire-and-forget task — but only after
`deliver_and_record` actually delivered the in-character «ща сфоткаю» ack, so
a pipeline failure never leaves a photo without its promise. The canonical
log line records the run as `action=replied+photo`.

## Pipeline error handling

Pipeline failures in `run_pipeline` are reported in chat only to users who
explicitly addressed the bot (an `@username` mention on a word boundary in
text/caption, or a reply to a bot message). Autonomous entry paths — random
media-response rolls, overheard insult checks and YouTube Shorts summaries —
fail silently: the error is logged as a warning and nothing is posted. (A
failed Shorts summary does get a canned «не смог посмотреть» reply when the
link's sender also explicitly addressed the bot.)

Additional rules:

- Quota/rate-limit notices (`DailyLimitError`, `RateLimitError`) are throttled
  to one full text notice per chat per 30 minutes; within the cooldown an
  addressed user gets a 😴 reaction instead of a repeated wall of text.
- `ContextLengthError` advice depends on how the message arrived: replies get
  "start a new message instead of replying to the old chain", non-replies get
  "the message itself is too long, shorten it".

## Stat columns tracked per media type

```
text        night_messages, link_messages, forwarded_messages, emoji_messages, long_message_max
voice       voice_messages, voice_max_duration
photo       photo_messages
sticker     sticker_messages
video       video_messages
animation   animation_messages
reactions   laugh_reactions, heart_reactions, fire_reactions, thumbsup_reactions
```
