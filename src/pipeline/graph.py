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
                                └─ ok → context_builder → intent_classifier
                                              ├─► worker_games ─┐
                                              ├─► worker_media ─┤
                                              └─► worker_general┘
                                                                └─► response → memory_writer → END
"""

from langgraph.graph import END, START, StateGraph

from src import config, log
from src.pipeline.context_builder import ContextBuilder
from src.pipeline.filter_node import MeaninglessFilterNode
from src.pipeline.guard_node import GuardNode
from src.pipeline.ingester import MessageIngester
from src.pipeline.intent_node import IntentClassifierNode
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


def route_by_intent(state: BotState) -> str:
    intent = state.get("intent") or "general"
    if intent == "games":
        return "worker_games"
    if intent == "media":
        return "worker_media"
    return "worker_general"


def build_pipeline(agent) -> StateGraph:
    """Build and compile the pipeline graph. Call once at startup."""
    graph = StateGraph(BotState)

    graph.add_node("router", MessageRouter(bot_username=config.BOT_USERNAME, bot_id=config.BOT_ID))
    graph.add_node("ingester", MessageIngester())
    graph.add_node("filter", MeaninglessFilterNode())
    graph.add_node("guard", GuardNode())
    graph.add_node("context_builder", ContextBuilder())
    graph.add_node("intent_classifier", IntentClassifierNode(agent))
    graph.add_node("worker_games", WorkerNode(agent, "games"))
    graph.add_node("worker_media", WorkerNode(agent, "media"))
    graph.add_node("worker_general", WorkerNode(agent, "general"))
    graph.add_node("response", ResponseNode(agent))
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
    graph.add_edge("context_builder", "intent_classifier")
    graph.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {"worker_games": "worker_games", "worker_media": "worker_media", "worker_general": "worker_general"},
    )
    graph.add_edge("worker_games", "response")
    graph.add_edge("worker_media", "response")
    graph.add_edge("worker_general", "response")
    graph.add_edge("response", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile()
