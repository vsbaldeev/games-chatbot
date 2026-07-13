"""HTTP client for the self-hosted image-generation service (imagegen-service/).

Mirrors the TTS never-raise contract: ``generate_image`` returns PNG bytes
or ``None`` on any failure, so a media failure can only demote a post,
never kill it. Generation takes minutes on the CPU host, so the service
exposes an async job API and this client polls it.
"""

import asyncio
import base64
import time

import httpx

from src import config, log

logger = log.get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 10


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
        response = await client.get(f"/generations/{generation_id}")
        if response.status_code == 404:
            logger.warning("imagegen job %s unknown or expired", generation_id)
            return None
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        if status == "done":
            return base64.b64decode(data["image_png_base64"])
        if status == "failed":
            logger.warning("imagegen job %s failed: %s", generation_id, data.get("error"))
            return None
    logger.warning("imagegen job %s missed the %ss deadline", generation_id, config.IMAGEGEN_DEADLINE_SECONDS)
    return None
