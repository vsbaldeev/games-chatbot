"""Scheduled job managers — each class registers a single job on the Telegram Application."""

import datetime
from abc import ABC, abstractmethod

from telegram.ext import Application

from src.jobs.agent import reset_model_job
from src.jobs.cleanup import cleanup_messages_job
from src.jobs.life_post import (
    CATCH_UP_DELAY_SECONDS as LIFE_POST_CATCH_UP_DELAY_SECONDS,
    LIFE_POST_RUN_TIME,
    catch_up_life_post_job,
    life_post_job,
)
from src.jobs.meme import daily_meme_job
from src.jobs.roles import CATCH_UP_DELAY_SECONDS, ROLES_RUN_TIME, catch_up_roles_job, weekly_roles_job
from src.jobs.ytdlp_update import ytdlp_update_job


class JobManagerInterface(ABC):
    @abstractmethod
    def add_jobs(self, app: Application) -> None: ...


class RolesJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(weekly_roles_job, time=ROLES_RUN_TIME)
        # Recover a Sunday run missed while the bot was down (e.g. network outage).
        app.job_queue.run_once(catch_up_roles_job, when=CATCH_UP_DELAY_SECONDS)


class LifePostJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(life_post_job, time=LIFE_POST_RUN_TIME)
        # Posts the deployment opener on a fresh install, or recovers a
        # missed scheduled slot (e.g. the bot was down during a planned post).
        app.job_queue.run_once(catch_up_life_post_job, when=LIFE_POST_CATCH_UP_DELAY_SECONDS)


class MemeJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(
            daily_meme_job,
            time=datetime.time(hour=15, minute=0, tzinfo=datetime.timezone.utc),
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


class YtdlpUpdateJobManager(JobManagerInterface):
    """Registers the daily yt-dlp freshness check (self-healing extractor rot)."""

    def add_jobs(self, app: Application) -> None:
        # 03:30 UTC: dead hours for the chat, after the cleanup job.
        app.job_queue.run_daily(
            ytdlp_update_job,
            time=datetime.time(hour=3, minute=30, tzinfo=datetime.timezone.utc),
        )
