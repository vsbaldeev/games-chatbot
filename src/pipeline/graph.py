"""
LangGraph wiring for the bot pipeline: a parent graph that runs two stage subgraphs.

  START → router
    ├─ should_respond (any media)  → ingest
    ├─ text + long + not forwarded → memory_writer
    └─ text + other                → END
  ingest  (ingester → filter nodes; see ``src.pipeline.ingest.graph``)
    ├─ should_respond=True  → respond
    └─ otherwise            → END
  respond (guard → context → worker → response; see ``src.pipeline.respond.graph``)
    ├─ blocked → END
    └─ otherwise → memory_writer → END

The router stays in the parent graph, as does memory_writer: the router's "unaddressed
long text → memory_writer" branch would be indistinguishable from a filter drop inside a
subgraph.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from telegram.ext import ContextTypes

from src import config, log
from src.pipeline.ingest.graph import build_ingest_graph, route_if_responding
from src.pipeline.memory_writer import MemoryWriter, MIN_PASSIVE_LENGTH
from src.pipeline.respond.graph import build_respond_graph
from src.pipeline.router import MessageRouter
from src.pipeline.state import BotState

logger = log.get_logger(__name__)

PIPELINE_BOT_DATA_KEY = "pipeline"


def route_after_router(state: BotState) -> str:
    msg = state["incoming"]
    if msg["media_type"] == "text":
        if state["should_respond"]:
            return "ingest"
        if not msg.get("is_forwarded") and len((msg.get("raw_text") or "").strip()) >= MIN_PASSIVE_LENGTH:
            return "memory_writer"
        return END
    return "ingest"


def route_after_respond(state: BotState) -> str:
    """End a blocked run; hand every other reply to the memory writer."""
    return END if state.get("blocked") else "memory_writer"


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
    graph.add_node("ingest", build_ingest_graph())
    graph.add_node("respond", build_respond_graph(worker_agent, response_agent))
    graph.add_node("memory_writer", MemoryWriter())

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"ingest": "ingest", "memory_writer": "memory_writer", END: END},
    )
    graph.add_conditional_edges("ingest", route_if_responding("respond"), {"respond": "respond", END: END})
    graph.add_conditional_edges(
        "respond", route_after_respond, {"memory_writer": "memory_writer", END: END}
    )
    graph.add_edge("memory_writer", END)

    return graph.compile()
