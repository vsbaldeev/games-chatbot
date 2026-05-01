"""Scheduled job: reset the agent's model rotation index."""

from src.agent import agent
from telegram.ext import ContextTypes


async def reset_model_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    await agent.reset_model_index()
