"""Pytest configuration for integration tests.

Loads real API keys from .env before any src.* import occurs.

IMPORTANT — isolation requirement:
    Run integration tests separately:  pytest tests/integration/
    NOT together with unit tests:      pytest tests/
    When run together, src.config is imported with unit-test stubs during
    collection and the integration conftest can no longer override the frozen
    config attributes.
"""

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

DOTENV_PATH = Path(__file__).parents[2] / ".env"

for key, value in dotenv_values(DOTENV_PATH).items():
    if value:
        os.environ[key] = value


@pytest.fixture
async def worker_agent():
    """Build and initialize a WorkerAgent backed by real Groq credentials.

    Returns:
        Initialized WorkerAgent ready for invoke_worker calls.
    """
    from src.agent.worker import WorkerAgent
    agent = WorkerAgent()
    await agent.init()
    return agent
