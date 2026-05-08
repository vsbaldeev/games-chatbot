"""
Telegram bot — entry point and startup lifecycle.
Run with: python -m src.bot
"""

from telegram import Update
from telegram.ext import Application, ApplicationBuilder

from src import achievements, log
from src.agent import agent
from src.bot.jobs import RoastJobManager, RolesJobManager, SilenceSweepJobManager, ResetModelJobManager, MessageCleanupJobManager
from src.memes import store as meme_store
from src.store import db as database, unified_messages as msg_store, user_memories as memory_store

log.setup()
logger = log.get_logger(__name__)


async def __on_startup(application: Application) -> None:
    await database.init()
    await agent.init()
    await achievements.init_tables()
    await msg_store.init_table()
    await memory_store.init_table()
    await meme_store.init_table()
    logger.info("Bot started, all tables and jobs initialized")


def main() -> None:
    from src import config
    from src.bot.handlers import (
        EventHandlerManager,
        CommandHandlerManager,
        MessageHandlerManager,
    )

    app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).post_init(__on_startup).build()

    for manager in [EventHandlerManager(), CommandHandlerManager(), MessageHandlerManager()]:
        manager.add_handlers(app)

    for job_manager in [RoastJobManager(), RolesJobManager(), SilenceSweepJobManager(), ResetModelJobManager(), MessageCleanupJobManager()]:
        job_manager.add_jobs(app)

    logger.info("Starting polling...")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
