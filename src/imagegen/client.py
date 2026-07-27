"""HTTP client for the self-hosted image-generation service (imagegen-service/).

Mirrors the TTS never-raise contract: ``generate_image`` returns PNG bytes
or ``None`` on any failure, so a media failure can only demote a post,
never kill it. Generation takes minutes on the CPU host, so the service
exposes an async job API and this client polls it.

Polling is deliberately patient and submitting deliberately is not. A
status read is free and repeatable, so transient read failures are retried
until the deadline rather than discarding a job that is still running. A
submit is neither: a request that timed out may already have created the
job server-side, and retrying it would burn a second three-minute
generation, so only a connect error — where no request can have landed — is
retried.
"""

import asyncio
import base64
import time

import httpx

from src import config, log

logger = log.get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 10

# Status reads get a longer timeout than the submit: while a generation runs,
# torch saturates every core on the host (TORCH_THREADS == the VPS's vCPU
# count), so uvicorn's event loop can be slow to answer even though the job is
# perfectly healthy. Polls are retried until the deadline regardless — the
# generous timeout just avoids treating a busy host as a blip at all.
POLL_TIMEOUT_SECONDS = 30

# HTTP status at or above which a poll is retried rather than given up on:
# 5xx is the service having a bad moment, not a verdict on the job.
TRANSIENT_STATUS_FLOOR = 500


class GenerationEnded(Exception):
    """A polled job reached a terminal state without producing an image.

    Raised for an expired/unknown id or a ``failed`` job — cases where
    further polling is pointless, as opposed to a transient read error.
    """


class TransientPollError(Exception):
    """One status read failed in a way that says nothing about the job.

    A 5xx from the service; the caller keeps polling until the deadline.
    """


async def generate_image(prompt: str) -> bytes | None:
    """Generate one image on the imagegen service.

    Args:
        prompt: Full generation prompt (character descriptor + scene).

    Returns:
        PNG bytes on success, or None when the service is disabled
        (``IMAGEGEN_URL`` empty), unreachable, the job failed, or the
        ``IMAGEGEN_DEADLINE_SECONDS`` deadline passed — never raises.
    """
    if not config.IMAGEGEN_URL:
        return None
    try:
        async with httpx.AsyncClient(
            base_url=config.IMAGEGEN_URL, timeout=REQUEST_TIMEOUT_SECONDS
        ) as client:
            generation_id = await submit_generation(client, prompt)
            if generation_id is None:
                return None
            return await poll_generation(client, generation_id)
    except Exception as error:
        logger.warning("Image generation failed: %s", error)
        return None


async def submit_generation(client: httpx.AsyncClient, prompt: str) -> str | None:
    """Submit a generation job, retrying once on a connect error.

    Args:
        client: HTTP client bound to the service base URL.
        prompt: Full generation prompt.

    Returns:
        The job's generation id, or None when both attempts failed to
        connect.
    """
    body = {
        "prompt": prompt,
        "width": config.IMAGEGEN_SIZE,
        "height": config.IMAGEGEN_SIZE,
        "steps": config.IMAGEGEN_STEPS,
        "guidance_scale": config.IMAGEGEN_GUIDANCE,
    }
    for attempt in range(2):
        try:
            response = await client.post("/generations", json=body)
            response.raise_for_status()
            return str(response.json()["generation_id"])
        except httpx.ConnectError as error:
            logger.warning("imagegen connect failed (attempt %d): %s", attempt + 1, error)
    return None


async def poll_generation(client: httpx.AsyncClient, generation_id: str) -> bytes | None:
    """Poll a submitted job until it finishes, fails or times out.

    A single failed status read must never discard a generation that is
    still running: on this host an image costs ~3 minutes of CPU, and there
    are dozens of polls per job, so one dropped connection or slow reply
    used to throw all that work away. Transient read errors are logged and
    retried until the deadline; only a terminal verdict from the service
    (expired id, ``failed`` job) stops the wait early.

    Args:
        client: HTTP client bound to the service base URL.
        generation_id: Id returned by :func:`submit_generation`.

    Returns:
        Decoded PNG bytes when the job reaches ``done``, or None on a
        ``failed``/expired job or when the deadline passes first.
    """
    deadline = time.monotonic() + config.IMAGEGEN_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        await asyncio.sleep(config.IMAGEGEN_POLL_SECONDS)
        try:
            png_bytes = await read_generation(client, generation_id)
        except (httpx.TransportError, TransientPollError) as error:
            logger.warning("imagegen poll for job %s failed, still waiting: %s", generation_id, error)
            continue
        except GenerationEnded as error:
            logger.warning("imagegen job %s ended without an image: %s", generation_id, error)
            return None
        if png_bytes is not None:
            return png_bytes
    logger.warning("imagegen job %s missed the %ss deadline", generation_id, config.IMAGEGEN_DEADLINE_SECONDS)
    return None


async def read_generation(client: httpx.AsyncClient, generation_id: str) -> bytes | None:
    """Read one status update for a submitted job.

    Args:
        client: HTTP client bound to the service base URL.
        generation_id: Id of the job to read.

    Returns:
        Decoded PNG bytes once the job is ``done``, or None while it is
        still queued or running.

    Raises:
        GenerationEnded: The job expired, is unknown, or failed.
        TransientPollError: The service answered 5xx.
        httpx.TransportError: The read timed out or the connection dropped.
        httpx.HTTPStatusError: Any other unexpected status — a contract
            mismatch worth surfacing rather than retrying for 20 minutes.
    """
    response = await client.get(f"/generations/{generation_id}", timeout=POLL_TIMEOUT_SECONDS)
    if response.status_code == 404:
        raise GenerationEnded("unknown or expired generation id")
    if response.status_code >= TRANSIENT_STATUS_FLOOR:
        raise TransientPollError(f"service returned HTTP {response.status_code}")
    response.raise_for_status()
    data = response.json()
    status = data.get("status")
    if status == "done":
        return base64.b64decode(data["image_png_base64"])
    if status == "failed":
        raise GenerationEnded(f"job failed: {data.get('error')}")
    return None
