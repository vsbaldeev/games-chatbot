"""
The ingest subgraph: turns the message into text, then decides whether the bot answers.

  START → ingester
            ├─ should_respond=False → END  (media described+stored, no reply)
            ├─ youtube_short / social_link → link_check ──────────────┐
            ├─ insult_check                → overheard ─┐             │
            ├─ media                       → media_check ┤→ budget ───┤→ END
            └─ text                        → classify ───┘             │
  a node that sets should_respond=False also ends the run

The four filter nodes (link_check, media_check, classify, budget) plus ``overheard`` are
added by ``add_filter_stage``. The subgraph sees only ``IngestState``.
"""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.pipeline.ingest.filter.budget import charge_budget
from src.pipeline.ingest.filter.classify import ClassifyNode
from src.pipeline.ingest.filter.link_check import check_link
from src.pipeline.ingest.filter.media_check import check_media
from src.pipeline.ingest.filter.overheard import OverheardNode
from src.pipeline.ingest.media.node import MessageIngester
from src.pipeline.state import BotState, IngestState


def route_after_ingester(state: BotState) -> str:
    """Pick the filter node that handles the message, or end when no reply is wanted."""
    if not state["should_respond"]:
        return END
    trigger = state.get("response_trigger")
    if trigger in ("youtube_short", "social_link"):
        return "link_check"
    if trigger == "insult_check":
        return "overheard"
    if state["incoming"]["media_type"] != "text":
        return "media_check"
    return "classify"


def route_if_responding(next_node: str) -> Callable[[BotState], str]:
    """Build a router that continues to ``next_node`` unless a node set should_respond=False."""

    def route(state: BotState) -> str:
        return next_node if state["should_respond"] else END

    return route


def add_filter_stage(
    graph: StateGraph,
    *,
    source: str,
    on_pass: str,
    classify_node: ClassifyNode | None = None,
    overheard_node: OverheardNode | None = None,
) -> None:
    """Add the filter nodes and their edges to ``graph``.

    Args:
        graph: The graph under construction.
        source: Node (or ``START``) whose output feeds the filter stage.
        on_pass: Node that follows when the message passes the filter.
        classify_node: Classifier to use; built from configuration when None
            (tests inject one with mock LLMs).
        overheard_node: Overheard-mention classifier; built from configuration
            when None.
    """
    graph.add_node("link_check", check_link)
    graph.add_node("media_check", check_media)
    graph.add_node("classify", classify_node if classify_node is not None else ClassifyNode())
    graph.add_node("overheard", overheard_node if overheard_node is not None else OverheardNode())
    graph.add_node("budget", charge_budget)

    graph.add_conditional_edges(
        source,
        route_after_ingester,
        {
            "link_check": "link_check",
            "overheard": "overheard",
            "media_check": "media_check",
            "classify": "classify",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "link_check", route_if_responding(on_pass), {on_pass: on_pass, END: END}
    )
    for checked_node in ("media_check", "classify", "overheard"):
        graph.add_conditional_edges(
            checked_node, route_if_responding("budget"), {"budget": "budget", END: END}
        )
    graph.add_conditional_edges("budget", route_if_responding(on_pass), {on_pass: on_pass, END: END})


def build_ingest_graph() -> CompiledStateGraph:
    """Build the ingest subgraph; its END hands the run back to the parent graph."""
    graph = StateGraph(IngestState)
    graph.add_node("ingester", MessageIngester())
    graph.add_edge(START, "ingester")
    add_filter_stage(graph, source="ingester", on_pass=END)
    return graph.compile()
