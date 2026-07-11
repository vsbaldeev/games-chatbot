"""
Telegram bot — entry point and startup lifecycle.
Run with: python -m src.bot
"""

from telegram import Update
from telegram.ext import Application, ApplicationBuilder

from src import log
from src.agent import worker_agent, response_agent, roast_agent, comedian_agent
from src.bot.jobs import (
    DailyActivityJobManager,
    LifePostJobManager,
    MemeJobManager,
    MessageCleanupJobManager,
    ResetModelJobManager,
    RolesJobManager,
    YtdlpUpdateJobManager,
)
from src.life.writer import episode_writer_agent
from src.store import db as database
from src.tts import speech_service

log.setup()
logger = log.get_logger(__name__)


async def __on_startup(application: Application) -> None:
    # The database schema is owned by Alembic migrations (`alembic upgrade head`),
    # applied before the bot process starts — the bot no longer creates tables.
    await database.init()
    await worker_agent.init()
    await response_agent.init()
    await roast_agent.init()
    await comedian_agent.init()
    await episode_writer_agent.init()
    await speech_service.init()
    logger.info("Bot started, all agents and jobs initialized")


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

    for job_manager in [RolesJobManager(), ResetModelJobManager(), MessageCleanupJobManager(), MemeJobManager(), YtdlpUpdateJobManager(), LifePostJobManager(), DailyActivityJobManager()]:
        job_manager.add_jobs(app)

    logger.info("Starting polling...")
    # bootstrap_retries=-1 lets the startup handshake (get_me/delete_webhook) retry
    # indefinitely with backoff instead of crashing on a transient network/DNS failure
    # — e.g. when the container starts before networking is ready. This mirrors the
    # polling loop, which already retries indefinitely.
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
        bootstrap_retries=-1,
    )
