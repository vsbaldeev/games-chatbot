"""Integration tests for WorkerAgent with the real Groq LLM API.

These tests consume Groq tokens. Run in isolation:
    pytest tests/integration/test_worker_agent.py

The worker_agent fixture (from conftest.py) initializes once per test function
using real credentials loaded from .env.
"""

import pytest

from src.agent.worker import WorkerAgent


@pytest.mark.integration
async def test_worker_agent_init_succeeds():
    """WorkerAgent.init() must complete without error when given real Groq credentials.

    Verifies that the API key is accepted and the executor is built successfully.
    No LLM call is made during init — it only constructs the LangChain agent.
    """
    agent = WorkerAgent()
    await agent.init()


@pytest.mark.integration
async def test_invoke_worker_returns_non_empty_text(worker_agent):
    """invoke_worker must return a non-empty string for a prompt that needs no tools.

    The date is embedded directly in the prompt so the model can answer without
    calling any external tool.
    """
    prompt = (
        "Current datetime: 2026-05-15 12:00 UTC\n\n"
        "Question from @testuser: Какое сегодня число?"
    )
    result, tools_used = await worker_agent.invoke_worker(prompt)
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.integration
async def test_invoke_worker_uses_context_before_tools(worker_agent):
    """invoke_worker must extract facts from the reply chain without calling tools.

    The CONTEXT FIRST directive in WORKER_PROMPT instructs the model to use
    information already present in the reply chain instead of reaching for tools.
    The release year is embedded in the prompt context, so the model must return
    "2022" without making any external API calls.
    """
    prompt = (
        "Context (reply chain):\n"
        "@alice: Elden Ring was released in February 2022.\n\n"
        "Question from @bob: Когда вышел Elden Ring?"
    )
    result, tools_used = await worker_agent.invoke_worker(prompt)
    assert "2022" in result
