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
                                     increment_stat → check_new → notify_unlocks

messages.py
    handle_message(update)         — text: track stats (night, link, forward, emoji, long),
                                     then enter the LangGraph pipeline
    handle_voice_message(update)   — increment voice_messages + update voice_max_duration,
                                     enter pipeline
    handle_photo_message(update)   — increment photo_messages, enter pipeline
    handle_sticker_message(update) — increment sticker_messages, enter pipeline
    handle_video_message(update)   — increment video_messages, enter pipeline
    handle_animation_message(update) — increment animation_messages (no pipeline)
```

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
