"""
The respond subgraph: builds the reply for a message the ingest stage let through.

  START → guard
            ├─ blocked → END
            ├─ meme / group-profile request → no_text_reply → END
            └─ ok → context_builder
                      ├─ needs facts            → worker ──────┐
                      └─ directive or link reply → skip_worker ┤
                                                               ▼
                                                   response ─(foreign script)→ language_correction → END
                                                      └─ clean → END

The subgraph sees only ``RespondState``. Storing what was learned (memory_writer) is the
parent graph's job, so a blocked message simply ends the run there.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.llm import needs_russian_correction
from src.pipeline.respond.context_builder import ContextBuilder
from src.pipeline.respond.guard_node import GuardNode
from src.pipeline.respond.reply.language_correction_node import LanguageCorrectionNode
from src.pipeline.respond.reply.response_node import ResponseNode, no_text_reply
from src.pipeline.respond.worker_node import WorkerNode, skip_worker, worker_is_skipped
from src.pipeline.state import BotState, RespondState


def route_by_guard(state: BotState) -> str:
    """End a blocked run; send meme and group-profile requests past the reply nodes."""
    if state.get("blocked"):
        return END
    if state.get("meme_request") or state.get("group_profile_request"):
        return "no_text_reply"
    return "context_builder"


def route_after_context(state: BotState) -> str:
    """Send replies that need no tool facts past the worker."""
    return "skip_worker" if worker_is_skipped(state) else "worker"


def route_after_response(state: BotState) -> str:
    """Route to language_correction when the response contains foreign script."""
    response = state.get("response") or ""
    if needs_russian_correction(response):
        return "language_correction"
    return END


def build_respond_graph(worker_agent, response_agent) -> CompiledStateGraph:
    """Build the respond subgraph; its END hands the run back to the parent graph."""
    graph = StateGraph(RespondState)

    graph.add_node("guard", GuardNode())
    graph.add_node("context_builder", ContextBuilder())
    graph.add_node("worker", WorkerNode(worker_agent))
    graph.add_node("skip_worker", skip_worker)
    graph.add_node("response", ResponseNode(response_agent))
    graph.add_node("no_text_reply", no_text_reply)
    graph.add_node("language_correction", LanguageCorrectionNode(response_agent))

    graph.add_edge(START, "guard")
    graph.add_conditional_edges(
        "guard",
        route_by_guard,
        {"context_builder": "context_builder", "no_text_reply": "no_text_reply", END: END},
    )
    graph.add_edge("no_text_reply", END)
    graph.add_conditional_edges(
        "context_builder", route_after_context, {"worker": "worker", "skip_worker": "skip_worker"}
    )
    graph.add_edge("worker", "response")
    graph.add_edge("skip_worker", "response")
    graph.add_conditional_edges(
        "response",
        route_after_response,
        {"language_correction": "language_correction", END: END},
    )
    graph.add_edge("language_correction", END)
    return graph.compile()
