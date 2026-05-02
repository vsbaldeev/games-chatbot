"""
LangGraph StateGraph wiring for the bot pipeline.

Graph edges:
  START → router_node
    ├─ should_respond=False → END
    └─ should_respond=True  → ingester_node
                                └─► guard_node
                                      ├─ blocked=True  → END
                                      └─ blocked=False → context_builder_node
                                                              └─► agent_node
                                                                      └─► memory_writer_node
                                                                                └─► END
"""

from langgraph.graph import END, START, StateGraph

from src import config, log
from src.pipeline.agent_node import AgentNode
from src.pipeline.context_builder import ContextBuilder
from src.pipeline.guard_node import GuardNode
from src.pipeline.ingester import MessageIngester
from src.pipeline.memory_writer import MemoryWriter
from src.pipeline.router import MessageRouter
from src.pipeline.state import BotState

logger = log.get_logger(__name__)


def route_by_response(state: BotState) -> str:
    return "ingester" if state["should_respond"] else END


def route_by_guard(state: BotState) -> str:
    return END if state.get("blocked") else "context_builder"


def build_pipeline(agent) -> StateGraph:
    """Build and compile the pipeline graph. Call once at startup."""
    router = MessageRouter(bot_username=config.BOT_USERNAME)
    ingester = MessageIngester()
    guard = GuardNode()
    context_builder = ContextBuilder()
    agent_node = AgentNode(agent)
    memory_writer = MemoryWriter()

    graph = StateGraph(BotState)
    graph.add_node("router", router)
    graph.add_node("ingester", ingester)
    graph.add_node("guard", guard)
    graph.add_node("context_builder", context_builder)
    graph.add_node("agent", agent_node)
    graph.add_node("memory_writer", memory_writer)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_by_response, {"ingester": "ingester", END: END})
    graph.add_edge("ingester", "guard")
    graph.add_conditional_edges("guard", route_by_guard, {"context_builder": "context_builder", END: END})
    graph.add_edge("context_builder", "agent")
    graph.add_edge("agent", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile()
