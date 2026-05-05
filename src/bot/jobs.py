"""Scheduled job managers — each class registers a single job on the Telegram Application."""

import datetime
from abc import ABC, abstractmethod

from telegram.ext import Application

from src.jobs.achievements import silence_sweep_job
from src.jobs.agent import reset_model_job
from src.jobs.roast import weekly_roast_job


class JobManagerInterface(ABC):
    @abstractmethod
    def add_jobs(self, app: Application) -> None: ...


class RoastJobManager(JobManagerInterface):
    def add_jobs(self, app: Application) -> None:
        app.job_queue.run_daily(
            weekly_roast_job,
            time=datetime.time(hour=12, minute=0, tzinfo=datetime.timezone.utc),
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
