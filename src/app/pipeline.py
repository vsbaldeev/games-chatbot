"""Pipeline manager — builds the LangGraph pipeline once and attaches it to the Telegram Application."""

from telegram.ext import Application

from src.agent import ResponseAgent, WorkerAgent
from src.pipeline.graph import PIPELINE_BOT_DATA_KEY, build_pipeline


class PipelineManager:
    def attach(self, app: Application) -> None:
        """Build the agents and the pipeline, exposing it to every callback as ``context.bot_data``."""
        app.bot_data[PIPELINE_BOT_DATA_KEY] = build_pipeline(WorkerAgent(), ResponseAgent())
