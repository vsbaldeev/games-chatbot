"""
LangGraph StateGraph wiring for the bot pipeline.

Graph edges:
  START → router
    ├─ photo (any) → ingester          ← photos always ingested so description
    ├─ should_respond=True → ingester    is stored for future reply-chain context
    └─ should_respond=False → END
                ingester
                    ├─ should_respond=True → guard
                    └─ should_respond=False → END  (photo described+stored, no reply)
                              guard
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
from src.pipeline.guard_node import GuardNode
from src.pipeline.ingester import MessageIngester
from src.pipeline.intent_node import IntentClassifierNode
from src.pipeline.memory_writer import MemoryWriter
from src.pipeline.response_node import ResponseNode
from src.pipeline.router import MessageRouter
from src.pipeline.state import BotState
from src.pipeline.worker_node import WorkerNode

logger = log.get_logger(__name__)


def route_after_router(state: BotState) -> str:
    # Photos always reach the ingester so the vision description is stored
    # in unified_messages even when the bot won't reply — this lets the bot
    # describe a photo later if someone replies to it and @mentions the bot.
    if state["incoming"]["media_type"] == "photo":
        return "ingester"
    return "ingester" if state["should_respond"] else END


def route_after_ingester(state: BotState) -> str:
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

    graph.add_node("router", MessageRouter(bot_username=config.BOT_USERNAME))
    graph.add_node("ingester", MessageIngester())
    graph.add_node("guard", GuardNode())
    graph.add_node("context_builder", ContextBuilder())
    graph.add_node("intent_classifier", IntentClassifierNode())
    graph.add_node("worker_games", WorkerNode(agent, "games"))
    graph.add_node("worker_media", WorkerNode(agent, "media"))
    graph.add_node("worker_general", WorkerNode(agent, "general"))
    graph.add_node("response", ResponseNode(agent))
    graph.add_node("memory_writer", MemoryWriter())

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {"ingester": "ingester", END: END})
    graph.add_conditional_edges("ingester", route_after_ingester, {"guard": "guard", END: END})
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
