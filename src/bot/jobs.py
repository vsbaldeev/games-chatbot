"""Scheduled job managers — each class registers a single job on the Telegram Application."""

import datetime
from abc import ABC, abstractmethod

from telegram.ext import Application

from src.jobs.achievements import silence_sweep_job
from src.jobs.agent import reset_model_job
from src.jobs.cleanup import cleanup_messages_job
from src.jobs.meme import daily_meme_job
from src.jobs.roles import CATCH_UP_DELAY_SECONDS, ROLES_RUN_TIME, catch_up_roles_job, weekly_roles_job


class JobManagerInterface(ABC):
    @abstractmethod
    def add_jobs(self, app: Application) -> None: ...


class RolesJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(weekly_roles_job, time=ROLES_RUN_TIME)
        # Recover a Sunday run missed while the bot was down (e.g. network outage).
        app.job_queue.run_once(catch_up_roles_job, when=CATCH_UP_DELAY_SECONDS)


class MemeJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(
            daily_meme_job,
            time=datetime.time(hour=15, minute=0, tzinfo=datetime.timezone.utc),
        )


class SilenceSweepJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(
            silence_sweep_job,
            time=datetime.time(hour=10, minute=0, tzinfo=datetime.timezone.utc),
        )


class ResetModelJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(
            reset_model_job,
            time=datetime.time(hour=0, minute=5, tzinfo=datetime.timezone.utc),
        )


class MessageCleanupJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(
            cleanup_messages_job,
            time=datetime.time(hour=3, minute=0, tzinfo=datetime.timezone.utc),
        )
