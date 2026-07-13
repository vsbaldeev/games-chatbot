"""In-memory generation-job registry with a single worker slot.

Named ``registry`` rather than the plan's ``queue``: a top-level module
called ``queue`` would shadow the stdlib module of the same name, which
torch imports internally — the service would crash on startup.

Jobs live in a dict and expire after ``JOB_TTL_SECONDS``; an
``asyncio.Semaphore(1)`` ensures one generation at a time (a second
concurrent diffusion run would blow the container's memory budget), and
the blocking generation itself runs on a thread so health checks stay
responsive.
"""

import asyncio
import base64
import dataclasses
import time
import uuid

from engine import Engine, GenerationParams

JOB_TTL_SECONDS = 3600


@dataclasses.dataclass
class Job:
    """One generation job's mutable state.

    Attributes:
        status: ``queued`` → ``running`` → ``done``/``failed``.
        created_at: Monotonic creation timestamp, drives expiry.
        image_png_base64: Base64 PNG once ``done``.
        error: Failure description once ``failed``.
    """

    status: str
    created_at: float
    image_png_base64: str | None = None
    error: str | None = None


class JobRegistry:
    """Holds jobs, runs them one at a time, and expires finished ones."""

    def __init__(self, engine: Engine) -> None:
        """Wire the registry to the generation engine.

        Args:
            engine: The engine that performs the actual generation.
        """
        self.__engine = engine
        self.__jobs: dict[str, Job] = {}
        self.__worker_slot = asyncio.Semaphore(1)

    def submit(self, params: GenerationParams) -> str:
        """Register a new job and schedule its execution.

        Args:
            params: Generation parameters for the job.

        Returns:
            The new job's id.
        """
        generation_id = uuid.uuid4().hex
        self.__jobs[generation_id] = Job(status="queued", created_at=time.monotonic())
        asyncio.get_running_loop().create_task(self.__run(generation_id, params))
        return generation_id

    def get(self, generation_id: str) -> Job | None:
        """Return a job by id, dropping expired jobs first.

        Args:
            generation_id: Id returned by :meth:`submit`.

        Returns:
            The job, or None when unknown or expired.
        """
        self.__expire()
        return self.__jobs.get(generation_id)

    async def __run(self, generation_id: str, params: GenerationParams) -> None:
        """Execute one job when the worker slot frees up.

        Args:
            generation_id: The job to execute.
            params: Its generation parameters.
        """
        async with self.__worker_slot:
            job = self.__jobs.get(generation_id)
            if job is None:
                return
            job.status = "running"
            try:
                png_bytes = await asyncio.to_thread(self.__engine.generate, params)
                job.image_png_base64 = base64.b64encode(png_bytes).decode()
                job.status = "done"
            except Exception as error:
                job.error = str(error)
                job.status = "failed"

    def __expire(self) -> None:
        """Drop jobs older than ``JOB_TTL_SECONDS``."""
        cutoff = time.monotonic() - JOB_TTL_SECONDS
        expired_ids = [
            job_id for job_id, job in self.__jobs.items() if job.created_at < cutoff
        ]
        for job_id in expired_ids:
            del self.__jobs[job_id]
