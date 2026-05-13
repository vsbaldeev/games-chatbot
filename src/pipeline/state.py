"""
Shared state types for the LangGraph pipeline.

BotState flows through every node:
  Router → Ingester → ContextBuilder → Agent → MemoryWriter
"""

from typing import Any

from typing_extensions import NotRequired, TypedDict


class IncomingMessage(TypedDict):
    """Raw and enriched data about the message that entered the pipeline."""

    update: Any                   # telegram.Update — not serialisable, kept as-is
    chat_id: int
    user_id: int
    username: str
    raw_text: str | None          # original message text or caption
    processed_text: str | None    # transcript / vision description, filled by Ingester
    media_type: str               # "text" | "voice" | "video_note" | "video" | "photo"
    message_id: int
    reply_to_msg_id: int | None
    file_id: str | None           # Telegram file_id for voice / photo messages
    is_forwarded: bool
    media_group_id: str | None    # Telegram media_group_id for album messages


class AssembledContext(TypedDict):
    """Everything the Agent node needs to build an enriched prompt."""

    user_facts: dict[str, list[str]]   # username → list of LLM-extracted fact strings
    recent_history: list[dict]         # flat recent messages used to fill the context window
    replied_to: dict | None            # the specific message being replied to, for annotation
    reply_chain: list[dict]            # full reply chain from root to replied-to message, oldest-first


class BotState(TypedDict):
    """Full mutable state passed between LangGraph nodes."""

    incoming: IncomingMessage
    should_respond: bool
    response_trigger: str          # "explicit" (@mention/reply) or "random" (25% chance)
    blocked: bool                  # True when Guard Node rejects the message
    context: AssembledContext | None
    response: str | None
    context_types: Any             # telegram.ext.ContextTypes instance for sending replies
    thread_id: NotRequired[str]    # derived from reply-chain root; scopes LLM history
    worker_output: NotRequired[str | None] # raw facts gathered by the worker
    search_notification_msg: NotRequired[Any]  # Telegram Message sent as search indicator
