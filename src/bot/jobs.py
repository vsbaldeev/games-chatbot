"""Scheduled jobs and their registration manager."""

import datetime
from abc import ABC, abstractmethod

from src import log
from telegram.error import BadRequest
from telegram.ext import Application, ContextTypes

from src import achievements
from src.agent import agent
from src.commands.fun import russian_roulette

logger = log.get_logger(__name__)


class JobManagerInterface(ABC):
    @abstractmethod
    def add_jobs(self, app: Application) -> None: ...


class ScheduledJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(
            russian_roulette,
            time=datetime.time(hour=18, minute=0, tzinfo=datetime.timezone.utc),
        )
        app.job_queue.run_daily(
            silence_sweep_job,
            time=datetime.time(hour=10, minute=0, tzinfo=datetime.timezone.utc),
        )
        app.job_queue.run_daily(
            reset_model_job,
            time=datetime.time(hour=0, minute=5, tzinfo=datetime.timezone.utc),
        )


async def silence_sweep_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = await achievements.get_all_chat_ids()
    for chat_id in chat_ids:
        members = await achievements.get_chat_members(chat_id)
        for user_id, username in members:
            try:
                new_ones = await achievements.check_silence_achievements(user_id, chat_id, username)
                for ach in new_ones:
                    text = (
                        f"🏆 @{username} получил достижение!\n\n"
                        f"{ach.emoji} *{ach.title}*\n_{ach.description}_"
                    )
                    try:
                        await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
                    except BadRequest:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=f"🏆 {username}: {ach.emoji} {ach.title} — {ach.description}",
                        )
            except Exception as error:
                logger.warning("Silence check failed for user %s in chat %s: %s", user_id, chat_id, error)


async def reset_model_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    await agent.reset_model_index()
