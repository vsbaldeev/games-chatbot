"""
LangGraph StateGraph wiring for the bot pipeline.

Graph edges:
  START → router
    ├─ text + should_respond=True  → ingester
    ├─ text + long + not forwarded → memory_writer
    ├─ text + other                → END
    └─ any media                  → ingester  (always, to enrich content in DB)
                ingester
                    ├─ should_respond=False → END  (media described+stored, no reply)
                    ├─ youtube_short / social_link → link_check ──┐
                    ├─ media                       → media_check ─┤→ budget
                    └─ text                        → classify ────┘
                link_check ─(should_respond)→ guard
                media_check, classify ─(should_respond)→ budget ─(should_respond)→ guard
                              guard
                                ├─ blocked → END
                                └─ ok → context_builder → worker → response
                                            ├─ foreign script → language_correction → memory_writer → END
                                            └─ ok             → memory_writer → END

The four filter nodes (link_check, media_check, classify, budget) are added by
``add_filter_stage``; a node that sets should_respond=False ends the run.
"""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from telegram.ext import ContextTypes

from src import config, log
from src.llm import needs_russian_correction
from src.pipeline.respond.context_builder import ContextBuilder
from src.pipeline.ingest.filter.budget import charge_budget
from src.pipeline.ingest.filter.classify import ClassifyNode
from src.pipeline.ingest.filter.link_check import check_link
from src.pipeline.ingest.filter.media_check import check_media
from src.pipeline.ingest.filter.overheard import OverheardNode
from src.pipeline.respond.guard_node import GuardNode
from src.pipeline.ingest.media.node import MessageIngester
from src.pipeline.respond.reply.language_correction_node import LanguageCorrectionNode
from src.pipeline.memory_writer import MemoryWriter, MIN_PASSIVE_LENGTH
from src.pipeline.respond.reply.response_node import ResponseNode, no_text_reply
from src.pipeline.router import MessageRouter
from src.pipeline.state import BotState
from src.pipeline.respond.worker_node import WorkerNode, skip_worker, worker_is_skipped

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
    return "memory_writer"


def add_filter_stage(
    graph: StateGraph,
    *,
    source: str,
    on_pass: str,
    classify_node: ClassifyNode | None = None,
    overheard_node: OverheardNode | None = None,
) -> None:
    """Add the four filter nodes and their edges to ``graph``.

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
    graph.add_node("guard", GuardNode())
    graph.add_node("context_builder", ContextBuilder())
    graph.add_node("worker", WorkerNode(worker_agent))
    graph.add_node("skip_worker", skip_worker)
    graph.add_node("response", ResponseNode(response_agent))
    graph.add_node("no_text_reply", no_text_reply)
    graph.add_node("language_correction", LanguageCorrectionNode(response_agent))
    graph.add_node("memory_writer", MemoryWriter())

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"ingester": "ingester", "memory_writer": "memory_writer", END: END},
    )
    add_filter_stage(graph, source="ingester", on_pass="guard")
    graph.add_conditional_edges(
        "guard",
        route_by_guard,
        {"context_builder": "context_builder", "no_text_reply": "no_text_reply", END: END},
    )
    graph.add_edge("no_text_reply", "memory_writer")
    graph.add_conditional_edges(
        "context_builder", route_after_context, {"worker": "worker", "skip_worker": "skip_worker"}
    )
    graph.add_edge("worker", "response")
    graph.add_edge("skip_worker", "response")
    graph.add_conditional_edges(
        "response",
        route_after_response,
        {"language_correction": "language_correction", "memory_writer": "memory_writer"},
    )
    graph.add_edge("language_correction", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile()
