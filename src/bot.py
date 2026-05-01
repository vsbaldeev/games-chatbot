"""
Entry point for the Telegram bot.
Run with: python -m src.bot
"""

import datetime

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    MessageReactionHandler,
    TypeHandler,
    filters,
)

from src import achievements, game_tracker, log
from src.agent import agent
from src import commands, dnd, duel, handlers, jobs, prozharka, roulette
from src.store import unified_messages as msg_store, user_memories as memory_store

log.setup()
logger = log.get_logger(__name__)


async def __reset_model_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    await agent.reset_model_index()


async def __on_startup(application: Application) -> None:
    await agent.init()
    await achievements.init_tables()
    await game_tracker.init_tables()
    await msg_store.init_table()
    await memory_store.init_table()

    application.job_queue.run_daily(
        roulette.russian_roulette,
        time=datetime.time(hour=18, minute=0, tzinfo=datetime.timezone.utc),
    )
    application.job_queue.run_daily(
        jobs.silence_sweep_job,
        time=datetime.time(hour=10, minute=0, tzinfo=datetime.timezone.utc),
    )
    application.job_queue.run_daily(
        __reset_model_job,
        time=datetime.time(hour=0, minute=5, tzinfo=datetime.timezone.utc),
    )
    logger.info("Bot started, all tables and jobs initialized")


def main() -> None:
    from src import config

    app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_TOKEN)
        .post_init(__on_startup)
        .build()
    )

    app.add_handler(TypeHandler(Update, handlers.track_member), group=-1)
    app.add_handler(
        MessageHandler(
            filters.StatusUpdate.NEW_CHAT_MEMBERS & filters.ChatType.GROUPS,
            handlers.handle_new_chat_members,
        )
    )
    app.add_handler(ChatMemberHandler(handlers.handle_bot_added_to_chat, ChatMemberHandler.MY_CHAT_MEMBER))

    app.add_handler(CommandHandler("start", commands.cmd_start, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("help", commands.cmd_help, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("multiplayer", commands.cmd_multiplayer, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("singleplayer", commands.cmd_singleplayer, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("achievements", commands.cmd_achievements, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("top", commands.cmd_top, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("roast", prozharka.cmd_roast, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("roulette", roulette.cmd_roulette, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("duel", duel.cmd_duel, filters=filters.ChatType.GROUPS))
    app.add_handler(CallbackQueryHandler(duel.handle_duel_callback, pattern=duel.DUEL_CALLBACK_PATTERN))
    app.add_handler(CommandHandler("dnd_pvp", dnd.cmd_dnd_pvp, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("dnd_coop", dnd.cmd_dnd_coop, filters=filters.ChatType.GROUPS))
    app.add_handler(CommandHandler("dnd_heist", dnd.cmd_dnd_heist, filters=filters.ChatType.GROUPS))
    app.add_handler(CallbackQueryHandler(dnd.handle_dnd_callback, pattern=dnd.DND_CALLBACK_PATTERN))

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
            handlers.handle_message,
        )
    )
    app.add_handler(
        MessageHandler(
            (filters.VOICE | filters.VIDEO_NOTE) & filters.ChatType.GROUPS,
            handlers.handle_voice_message,
        )
    )
    app.add_handler(
        MessageHandler(
            filters.PHOTO & filters.ChatType.GROUPS,
            handlers.handle_photo_message,
        )
    )
    app.add_handler(
        MessageHandler(
            filters.Sticker.ALL & filters.ChatType.GROUPS,
            handlers.handle_sticker_message,
        )
    )
    app.add_handler(
        MessageHandler(
            filters.VIDEO & filters.ChatType.GROUPS,
            handlers.handle_video_message,
        )
    )
    app.add_handler(MessageReactionHandler(
        handlers.handle_reaction,
        message_reaction_types=MessageReactionHandler.MESSAGE_REACTION_UPDATED,
    ))

    logger.info("Starting polling...")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
