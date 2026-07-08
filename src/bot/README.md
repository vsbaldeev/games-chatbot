Telegram Application wiring: handler registration, scheduled job setup, and startup lifecycle.

bot/__init__.py builds the Application, registers all HandlerManagers and JobManagers,
opens the DB connection pool and initialises the LLM agents, and calls
application.run_polling(). The database schema is provisioned separately by Alembic
migrations (`alembic upgrade head`) before the process starts — the bot no longer
creates tables.

## Handler managers

```
EventHandlerManager
    TypeHandler(Update, track_member)           — register every active user in chat_members
    MessageHandler(new_chat_members)            — greet new members
    ChatMemberHandler(bot_added)                — handle bot being added to a new group
    MessageReactionHandler(handle_reaction)     — track emoji reactions → user_stats

CommandHandlerManager
    /start          — welcome message
    /help           — command list
    /duel           — emoji duel picker
    /meme           — random image meme from Reddit
    CallbackQueryHandler(duel_*)   — duel inline buttons

MessageHandlerManager
    text        → handle_message        — main pipeline entry point
    voice       → handle_voice_message
    video_note  → handle_voice_message  (same handler, different media_type)
    photo       → handle_photo_message
    sticker     → handle_sticker_message
    video       → handle_video_message
    animation   → handle_animation_message
```

## Scheduled jobs

```
RolesJobManager          daily 14:00 UTC   weekly_roles_job        (exits early unless Sunday)
MemeJobManager           daily 15:00 UTC   daily_meme_job          (sends one fresh meme per chat)
ResetModelJobManager     daily 00:05 UTC   reset_model_job         (resets LLM fallback index to 0)
MessageCleanupJobManager daily 03:00 UTC   cleanup_messages_job    (prunes unified_messages and thread_history, 60-day retention)
YtdlpUpdateJobManager    daily 03:30 UTC   ytdlp_update_job        (installs newer yt-dlp into /app/runtime-deps and restarts the bot gracefully)
```
