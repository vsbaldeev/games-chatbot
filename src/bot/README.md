Telegram Application wiring: handler registration, scheduled job setup, and startup lifecycle.

bot/__init__.py builds the Application, registers all HandlerManagers and JobManagers,
initialises DB tables, and calls application.run_polling().

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
    /achievements   — last 3 earned achievements with total count
    /top            — top-3 leaderboard by achievement count
    /duel           — emoji duel picker
    /dnd_pvp        — D&D PvP adventure
    /dnd_coop       — D&D co-op adventure
    /dnd_heist      — D&D heist
    /roast          — on-demand прожарка
    CallbackQueryHandler(duel_*)   — duel inline buttons
    CallbackQueryHandler(dnd_*)    — D&D lobby/action buttons

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
RoastJobManager        daily 12:00 UTC   weekly_roast_job    (exits early unless it is this week's roast day)
RolesJobManager        daily 14:00 UTC   weekly_roles_job    (exits early unless Sunday)
SilenceSweepJobManager daily 10:00 UTC   silence_sweep_job   (awards silence achievements)
ResetModelJobManager   daily 00:05 UTC   reset_model_job     (resets LLM fallback index to 0)
```
