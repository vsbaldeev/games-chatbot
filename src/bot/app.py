"""
Telegram bot — entry point and startup lifecycle.
Run with: python -m src.bot
"""

from telegram import Update
from telegram.ext import Application, ApplicationBuilder, ContextTypes

from src import log
from src.agent import worker_agent, response_agent, roast_agent
from src.bot.jobs import (
    MemeJobManager,
    MessageCleanupJobManager,
    ResetModelJobManager,
    RolesJobManager,
    YtdlpUpdateJobManager,
)
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
    await speech_service.init()
    logger.info("Bot started, all agents and jobs initialized")


async def __on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log unhandled errors instead of letting PTB report them as "no error handlers".

    Args:
        update: The update that triggered the error, if any. Network errors from the
            polling loop (already retried indefinitely by PTB) pass ``None`` here.
        context: The callback context carrying the raised error in ``context.error``.
    """
    logger.warning("Unhandled error from PTB (auto-retried if network-related): %s", context.error)


def main() -> None:
    from src import config
    from src.bot.handlers import (
        EventHandlerManager,
        CommandHandlerManager,
        MessageHandlerManager,
    )

    app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).post_init(__on_startup).build()
    app.add_error_handler(__on_error)

    for manager in [EventHandlerManager(), CommandHandlerManager(), MessageHandlerManager()]:
        manager.add_handlers(app)

    job_managers = (
        RolesJobManager(),
        ResetModelJobManager(),
        MessageCleanupJobManager(),
        MemeJobManager(),
        YtdlpUpdateJobManager(),
    )
    for job_manager in job_managers:
        job_manager.add_jobs(app)

    logger.info("Starting polling...")


    app.run_polling(
    # drop_pending_updates=True discards updates queued while the bot was down (deploy,
    # crash, VPS reboot) so it doesn't process a stale backlog on restart — messages and
    # commands sent during downtime would otherwise fire late and confuse the pipeline.
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    # bootstrap_retries=-1 lets the startup handshake (get_me/delete_webhook) retry
    # indefinitely with backoff instead of crashing on a transient network/DNS failure
    # — e.g. when the container starts before networking is ready. This mirrors the
    # polling loop, which already retries indefinitely.
        bootstrap_retries=-1,
    )
