"""FastAPI app for self-hosted CPU image generation.

Async job API: generation takes minutes on a 4 vCPU host, so a POST
returns a job id immediately and the client polls. See README.md for the
full contract.
"""

import asyncio
import contextlib
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine import Engine, GenerationParams
from registry import JobRegistry

IDLE_CHECK_SECONDS = 60

logger = logging.getLogger("imagegen")
engine = Engine()
registry = JobRegistry(engine)


class GenerationRequest(BaseModel):
    """Body of ``POST /generations``.

    Attributes:
        prompt: Full positive prompt.
        negative_prompt: Optional negative prompt.
        width: Output width in pixels.
        height: Output height in pixels.
        steps: Diffusion steps (~20 with DPM++ 2M Karras).
        guidance_scale: CFG scale (~6 for standard sampling).
        seed: Optional seed for reproducible output.
    """

    prompt: str
    negative_prompt: str | None = None
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 6.0
    seed: int | None = None


async def idle_unload_loop() -> None:
    """Periodically release the pipeline after an idle stretch."""
    while True:
        await asyncio.sleep(IDLE_CHECK_SECONDS)
        unloaded = await asyncio.to_thread(engine.unload_if_idle)
        if unloaded:
            logger.info("Pipeline unloaded after idle period")


@contextlib.asynccontextmanager
async def lifespan(application: FastAPI):
    """Run the idle-unload watchdog for the app's lifetime.

    Args:
        application: The FastAPI application being started.

    Yields:
        Control to the running application.
    """
    watchdog = asyncio.get_running_loop().create_task(idle_unload_loop())
    yield
    watchdog.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    """Report service liveness and whether the model is resident.

    Returns:
        Status payload with the ``model_loaded`` flag.
    """
    return {"status": "ok", "model_loaded": engine.model_loaded}


@app.post("/generations", status_code=202)
async def create_generation(request: GenerationRequest) -> dict:
    """Accept a generation job and return its id immediately.

    Args:
        request: Generation parameters.

    Returns:
        Payload with the ``generation_id`` to poll.
    """
    params = GenerationParams(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
    )
    return {"generation_id": registry.submit(params)}


@app.get("/generations/{generation_id}")
async def get_generation(generation_id: str) -> dict:
    """Return a job's status and, once done, its image.

    Args:
        generation_id: Id returned by ``POST /generations``.

    Returns:
        Status payload; includes ``image_png_base64`` when ``done`` and
        ``error`` when ``failed``.

    Raises:
        HTTPException: 404 when the id is unknown or expired.
    """
    job = registry.get(generation_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired generation id")
    return {
        "status": job.status,
        "image_png_base64": job.image_png_base64,
        "error": job.error,
    }
