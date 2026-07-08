"""
LangGraph StateGraph wiring for the bot pipeline.

Graph edges:
  START → router
    ├─ text + should_respond=True  → ingester
    ├─ text + humor gate fires     → humor → memory_writer  (autonomous joke)
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

from src import config, log
from src.agent import comedian_agent, needs_russian_correction
from src.pipeline import humor_gate
from src.pipeline.context_builder import ContextBuilder
from src.pipeline.humor_node import HumorNode
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


def route_after_router(state: BotState) -> str:
    msg = state["incoming"]
    if msg["media_type"] == "text":
        if state["should_respond"]:
            return "ingester"
        if humor_gate.should_consider(msg["chat_id"], msg):
            return "humor"
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


def build_pipeline(worker_agent, response_agent) -> StateGraph:
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
    graph.add_node("humor", HumorNode(comedian_agent))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"ingester": "ingester", "humor": "humor", "memory_writer": "memory_writer", END: END},
    )
    graph.add_edge("humor", "memory_writer")
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
