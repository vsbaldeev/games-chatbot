"""
LangGraph StateGraph wiring for the bot pipeline.

Graph edges:
  START → router
    ├─ text + should_respond=True  → ingester
    ├─ text + long + not forwarded → memory_writer
    ├─ text + other                → END
    └─ any media                  → ingester  (always, to enrich content in DB)
                ingester
                    ├─ should_respond=True  → filter
                    └─ should_respond=False → END  (media described+stored, no reply)
                              filter
                                ├─ blocked → END
                                └─ ok → context_builder → worker → response
                                            ├─ foreign script → language_correction → memory_writer → END
                                            └─ ok             → memory_writer → END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from telegram.ext import ContextTypes

from src import config, log
from src.llm import needs_russian_correction
from src.pipeline.context_builder import ContextBuilder
from src.pipeline.filter_node import MeaninglessFilterNode
from src.pipeline.guard_node import GuardNode
from src.pipeline.ingester import MessageIngester
from src.pipeline.language_correction_node import LanguageCorrectionNode
from src.pipeline.memory_writer import MemoryWriter, MIN_PASSIVE_LENGTH
from src.pipeline.response_node import ResponseNode
from src.pipeline.router import MessageRouter
from src.pipeline.state import BotState
from src.pipeline.worker_node import WorkerNode

logger = log.get_logger(__name__)

PIPELINE_BOT_DATA_KEY = "pipeline"


def route_after_router(state: BotState) -> str:
    msg = state["incoming"]
    if msg["media_type"] == "text":
        if state["should_respond"]:
            return "ingester"
        if not msg.get("is_forwarded") and len((msg.get("raw_text") or "").strip()) >= MIN_PASSIVE_LENGTH:
            return "memory_writer"
        return END
    return "ingester"


def route_after_ingester(state: BotState) -> str:
    return "filter" if state["should_respond"] else END


def route_after_filter(state: BotState) -> str:
    return "guard" if state["should_respond"] else END


def route_by_guard(state: BotState) -> str:
    return END if state.get("blocked") else "context_builder"


def route_after_response(state: BotState) -> str:
    """Route to language_correction when the response contains foreign script."""
    response = state.get("response") or ""
    if needs_russian_correction(response):
        return "language_correction"
    return "memory_writer"


def get_pipeline(context: ContextTypes.DEFAULT_TYPE) -> CompiledStateGraph:
    """Return the compiled pipeline that ``src.app`` attached to the application.

    Args:
        context: The handler or job callback context.

    Returns:
        The graph built once at startup by ``PipelineManager.attach``.
    """
    return context.bot_data[PIPELINE_BOT_DATA_KEY]


def build_pipeline(worker_agent, response_agent) -> CompiledStateGraph:
    """Build and compile the pipeline graph. Call once at startup."""
    graph = StateGraph(BotState)

    graph.add_node("router", MessageRouter(bot_username=config.BOT_USERNAME, bot_id=config.BOT_ID))
    graph.add_node("ingester", MessageIngester())
    graph.add_node("filter", MeaninglessFilterNode())
    graph.add_node("guard", GuardNode())
    graph.add_node("context_builder", ContextBuilder())
    graph.add_node("worker", WorkerNode(worker_agent))
    graph.add_node("response", ResponseNode(response_agent))
    graph.add_node("language_correction", LanguageCorrectionNode(response_agent))
    graph.add_node("memory_writer", MemoryWriter())

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"ingester": "ingester", "memory_writer": "memory_writer", END: END},
    )
    graph.add_conditional_edges("ingester", route_after_ingester, {"filter": "filter", END: END})
    graph.add_conditional_edges("filter", route_after_filter, {"guard": "guard", END: END})
    graph.add_conditional_edges("guard", route_by_guard, {"context_builder": "context_builder", END: END})
    graph.add_edge("context_builder", "worker")
    graph.add_edge("worker", "response")
    graph.add_conditional_edges(
        "response",
        route_after_response,
        {"language_correction": "language_correction", "memory_writer": "memory_writer"},
    )
    graph.add_edge("language_correction", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile()
