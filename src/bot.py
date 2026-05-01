"""
Entry point for the Telegram bot.
Run with: python -m src.bot
"""

import datetime

from telegram import Update
from telegram.ext import Application, ApplicationBuilder, ContextTypes

from src import achievements, game_tracker, log
from src.agent import agent
from src import jobs, roulette
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
    from src.handlers.registry import (
        EventHandlerRegistry,
        CommandHandlerRegistry,
        MessageHandlerRegistry,
    )

    app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_TOKEN)
        .post_init(__on_startup)
        .build()
    )

    for registry in [EventHandlerRegistry(), CommandHandlerRegistry(), MessageHandlerRegistry()]:
        registry.register(app)

    logger.info("Starting polling...")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
