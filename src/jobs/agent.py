"""Scheduled job: reset the agent's model rotation index."""

from src.agent import worker_agent, response_agent, roast_agent, comedian_agent
from telegram.ext import ContextTypes


async def reset_model_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset every agent's model rotation index on schedule.

    Args:
        context: Telegram job context (unused).
    """
    await worker_agent.reset_model_index()
    await response_agent.reset_model_index()
    await roast_agent.reset_model_index()
    await comedian_agent.reset_model_index()
