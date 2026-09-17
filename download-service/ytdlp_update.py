"""Daily yt-dlp freshness check — self-heals extractor rot for this service only.

YouTube and Instagram break yt-dlp's extractors several times a year; a
stale extractor silently kills downloads until a new version is installed.
This job keeps the fix fully automatic and scoped to this container:

  1. ``entrypoint.sh`` upgrades yt-dlp into ``/service/runtime-deps`` (a
     writable overlay that shadows the baked package via ``PYTHONPATH``)
     on every container start.
  2. This job checks PyPI daily; when a newer yt-dlp exists it installs it
     and stops the process gracefully (SIGTERM). Docker's
     ``restart: unless-stopped`` brings the container back up, and the
     entrypoint's fresh copy takes effect. Only this service restarts —
     the bot process is never touched.

Worst case, a YouTube/Instagram-side breakage lasts about a day with no
human involved. Any error here is logged and skipped until the next run.

Installs with the same extras as ``requirements.txt``'s yt-dlp line and
``entrypoint.sh``'s startup install (``curl-cffi`` for Instagram, ``default``
for yt-dlp-ejs — yt-dlp's own JS challenge-solver scripts, exact-pinned by
yt-dlp to match its own version). A bare ``yt-dlp`` install here would
upgrade yt-dlp without its matching yt-dlp-ejs, silently breaking
JS-challenge-dependent formats the moment the two drift apart.
"""

import asyncio
import importlib.metadata
import logging
import os
import signal
import sys

logger = logging.getLogger("download-service.ytdlp_update")

RUNTIME_DEPS_DIR = "/service/runtime-deps"
PIP_TIMEOUT_SECONDS = 300
YTDLP_PACKAGE_SPEC = "yt-dlp[curl-cffi,default]"


async def run_pip(*pip_args: str) -> tuple[int, str]:
    """Run a pip command in a subprocess and capture its output.

    Args:
        pip_args: Arguments passed to ``python -m pip``.

    Returns:
        Tuple of the process return code and its combined stdout output.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pip", *pip_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        # stderr is merged into stdout, so the second element is always None.
        stdout, stderr_ignored = await asyncio.wait_for(
            process.communicate(), timeout=PIP_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        process.kill()
        return 1, "pip timed out"
    return process.returncode or 0, stdout.decode(errors="replace")


def newer_version_available(dry_run_output: str, current_version: str) -> bool:
    """Decide whether pip's dry-run output announces a newer yt-dlp.

    Args:
        dry_run_output: Combined output of ``pip install --dry-run``.
        current_version: The yt-dlp version currently imported.

    Returns:
        True when pip would install a version different from the current one.
    """
    for line in dry_run_output.splitlines():
        if line.strip().startswith("Would install") and "yt-dlp" in line:
            return current_version not in line
    return False


async def check_and_update() -> None:
    """Install a newer yt-dlp when one exists and restart this service.

    Returns:
        None. Any error is logged and skipped until the next daily run.
    """
    if not os.path.isdir(RUNTIME_DEPS_DIR):
        logger.info("yt-dlp update: %s absent (not in container), skipping", RUNTIME_DEPS_DIR)
        return
    try:
        current_version = importlib.metadata.version("yt-dlp")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("yt-dlp update: package not installed, skipping check")
        return
    return_code, output = await run_pip(
        "install", "--dry-run", "--upgrade", "--target", RUNTIME_DEPS_DIR,
        "--no-cache-dir", YTDLP_PACKAGE_SPEC,
    )
    if return_code != 0:
        logger.warning("yt-dlp update: dry-run failed, skipping until tomorrow: %s", output[-500:])
        return
    if not newer_version_available(output, current_version):
        logger.info("yt-dlp update: %s is current, nothing to do", current_version)
        return
    return_code, output = await run_pip(
        "install", "--upgrade", "--target", RUNTIME_DEPS_DIR, "--no-cache-dir", YTDLP_PACKAGE_SPEC,
    )
    if return_code != 0:
        logger.warning("yt-dlp update: install failed, skipping until tomorrow: %s", output[-500:])
        return
    logger.info(
        "yt-dlp update: newer version installed (was %s) — restarting for it to take effect",
        current_version,
    )
    # SIGTERM lets uvicorn shut down gracefully; docker's restart policy
    # brings the container back on the fresh version.
    os.kill(os.getpid(), signal.SIGTERM)
