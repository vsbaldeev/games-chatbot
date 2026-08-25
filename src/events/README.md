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

messages.py (deliver_response)
    deliver_response(final_state, msg, clean) — dispatches the pipeline's
                                      reply. A link trigger (youtube_short /
                                      social_link) whose fetch produced
                                      content is handed off to
                                      link_repost.deliver_link_message;
                                      voice/video_note triggers try a
                                      synthesized voice reply first
                                      (voice_reply.py, falling back to text);
                                      everything else is a plain text reply

link_repost.py
    deliver_link_message(msg, summary, video, username, url, is_bare,
                          cap_remaining) —
                                      sends the bot's ONE combined message
                                      for a link trigger: the downloaded
                                      video with the summary as its caption
                                      when there is one (YouTube Shorts,
                                      Instagram Reel), or a plain text
                                      message when there isn't (long-form
                                      YouTube never carries video, and a
                                      failed download degrades the same way)
```

When the triggering message was nothing but the link (`link_message_is_bare`,
set by the router — see src/pipeline/README.md), the message is sent
un-anchored and `build_caption` credits the sender ("Скинул @username" + the
canonical link + the summary), then `try_delete_original` removes the user's
original link message — only after the send already succeeded, never before,
and never at all if the send failed. Deleting degrades quietly: it needs the
bot to be a chat administrator with `can_delete_messages`, and a missing
permission just logs a warning and leaves the link message in place —
everything else about the feature still works. When the message carried more
than the link, it is left alone and the bot's single message replies to it
instead, with the bare summary only (no credit line, no link — both are
already visible in the original message). A bare link with no resolvable
canonical URL (`resolve_bare_deletion`) is never deleted regardless of the
bareness flag — deleting it would strand the link nowhere in the chat, so
the original is kept and the bot's reply carries the bare summary, same as
any other non-bare send.

If the combined send itself fails for any reason, `deliver_link_message` falls
back to an ordinary anchored text reply (`msg.reply_text`) and never deletes
the original — the download and LLM spend already happened, so the summary
still reaches the chat. The one exception is a video-less, non-bare send that
failed: that call already *was* the anchored text reply, so it re-raises
instead of retrying itself.

`fit_caption` fits the text inside the relevant Telegram limit through a
three-rung ladder: send as composed when it already fits; otherwise compress
the summary against the remaining budget via `src/agent/compress.py`
(LLM-based compression); and only if the compressor still overshoots,
`truncate_at_sentence` cuts it deterministically at the last sentence
boundary inside the budget (or appends an ellipsis when none exists) as a
last-resort backstop that can never fail to fit. The limit itself depends on
whether a video is attached: 1024 characters (Telegram's caption cap) when
there is one, 4096 (the plain text-message cap) when there isn't — long-form
YouTube links and any download failure that falls back to a text-only send
are not bound by the tighter caption limit.

When the router (`src/pipeline/router.py`) computed a `social_link_cap_remaining`
for the matched handler (Instagram only today — see `src.pipeline.social_links`),
`build_cap_remaining_footer` renders it as a trailing "Осталось рилсов сегодня: N"
line, reserved out of the compression budget the same way the credit line is —
it is never compressed or truncated away. Unlike the credit line, it is shown
regardless of `is_bare`: it survives even when the original message is kept and
the summary replies to it directly. A handler with no daily cap
(`daily_cap is None`, e.g. the long-form YouTube handler) never sets this field,
so no footer is shown for it.

The content block computed for the caption (transcript/frame
descriptions/comments for a Short, description/comments for a Reel or
long-form YouTube link) is also persisted on the bot's own message row
(`link_material` on `unified_messages`, written by `deliver_and_record` in
`messages.py`). When a member replies directly to that message,
`response_node.py` injects it back into the prompt with a no-fabrication
instruction, so the reply can be as specific as the original caption was —
see `docs/superpowers/specs/2026-08-12-link-reply-grounding-design.md`. The
`link_material` column marks the row for `MessageRouter`'s addressing gate:
a reply to a link-repost message only counts as addressing the bot when it
also mentions the bot or reads as a genuine question or request, unlike
every other bot message.

The bot's group-wide announcements (weekly roles and group profiles) are
similarly marked with `is_broadcast=True` (set by `send_and_store` when the
message is a broadcast), so replies to them can be treated as chat among
members rather than necessarily as messages to the bot. Unlike the
link-repost gate, this one is not decided in the router: a bare reply
(no @mention) still routes as "explicit" but carries `broadcast_reply`, and
`MeaninglessFilterNode` (`src/pipeline/filter_node.py`) is the one that may
silence it — an LLM `NOT_ADDRESSED` verdict, itself overridden back to a
reply whenever the text looks like a genuine question or request (the same
deterministic floor MEANINGLESS/BANTER get). An @mention skips this whole
path: it clears `broadcast_reply` at the router, so the bot is always
addressed unambiguously.

## Chat-requested selfies

When the pipeline accepted a photo request (`photo_request` state flag set by
the filter, no selfie already in flight), `run_pipeline` launches
`src/life/selfie.deliver_selfie` as a fire-and-forget task — but only after
`deliver_and_record` actually delivered the in-character «ща сфоткаю» ack, so
a pipeline failure never leaves a photo without its promise. The canonical
log line records the run as `action=replied+photo`.

## Chat-requested memes

When the pipeline accepted a meme request (`meme_request` state flag set by the
filter), the response node returns an **empty** reply, so `run_pipeline` falls
past the `if response.strip():` branch into the media-only dispatch — the one
path that answers with an image and no text. It fire-and-forgets
`deliver_meme`, and the canonical log line records `action=meme`.

`deliver_meme` sends an `upload_photo` chat action first: with no text reply,
that indicator is the only sign of life during the download and up to three
vision calls. If `memes.sender.send_meme` returns False — pool exhausted,
every candidate rejected, or the fail-closed vision gate unavailable — an
honest canned line from `MEME_FAILED_REPLIES` goes out instead. That fallback
is load-bearing: the `/meme` command was retired, so silence here would leave
a direct request unanswered with no other way to ask.

Like the selfie path, this is fire-and-forget, so the canonical log line emits
before the meme actually lands.

## Chat-requested group profiling

When the pipeline accepted a group-profile request (`group_profile_request`
state flag set by the filter — see `src/group_profile/README.md`), the
response node returns an **empty** reply, so `run_pipeline` falls into the
same media-only dispatch branch the meme path established, checked right
after it. It fire-and-forgets `deliver_group_profile`, and the canonical log
line records `action=profile`.

`deliver_group_profile` sends a `typing` chat action first — the only sign of
life while every chat member's dossier is gathered and the LLM call against
the user's own rubric runs — then calls `src.group_profile.profile.run_group_profile`.
If that returns `None` (no members at all) or raises, an honest canned line
from `GROUP_PROFILE_FAILED_REPLIES` goes out instead, same principle as the
meme path's fallback: a direct request must never end in silence.

Like the meme and selfie paths, this is fire-and-forget, so the canonical log
line emits before the profile message actually lands.

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
- If the worker already sent the «🔍 Ищу…» search-notification message this
  turn before a later node failed, the failure notice edits that message in
  place (`edit_and_store`, bypassing the cooldown's reaction fallback) instead
  of sending a second, separate message — `run_pipeline` streams the graph
  (`PIPELINE.astream(..., stream_mode="values")`) rather than using a plain
  `.ainvoke()` specifically so `final_state` still holds
  `search_notification_msg` when a later node raises. A worker-side failure
  that happens after the notification already fired but before the node
  returns (the exception itself, not the graph state) carries the same
  message via `err.search_notification_msg`, set in `WorkerNode.__call__`.

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
