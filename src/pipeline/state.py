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
    replied_to_fallback: NotRequired[dict | None]  # row-shaped copy of msg.reply_to_message, used when the DB row is missing; read-side only, never inserted


class AssembledContext(TypedDict):
    """Everything the Agent node needs to build an enriched prompt."""

    user_facts: dict[str, list[str]]   # username → list of LLM-extracted fact strings
    recent_history: list[dict]         # flat recent messages used to fill the context window
    replied_to: dict | None            # the specific message being replied to, for annotation
    reply_chain: list[dict]            # full reply chain from root to replied-to message, oldest-first
    asking_user_tag: dict | None       # {"tag", "reason"} weekly role of the message sender, if any
    mentioned_tags: dict[str, dict]    # username → {"tag", "reason"} for members @mentioned in the question
    bot_self_facts: list[str]          # canon facts about the bot's own life, relevant to this message
    bot_self_episodes: list[str]       # past life-post episodes relevant to this message
    bot_current_activity: tuple[str, str] | None  # (phrase, "fresh"|"recent") from the newest life post, or None
    bot_recent_activities: list[tuple[str, float]]  # (phrase, posted_at) history, newest first, for dated "what did you do" answers


class BotState(TypedDict):
    """Full mutable state passed between LangGraph nodes."""

    incoming: IncomingMessage
    should_respond: bool
    response_trigger: str          # "explicit" (@mention/reply), "insult_check" (bot-word mention), "random" (10% chance), "youtube_short" (Shorts link) or "humor" (autonomous joke)
    blocked: bool                  # True when Guard Node rejects the message
    context: AssembledContext | None
    response: str | None
    context_types: Any             # telegram.ext.ContextTypes instance for sending replies
    thread_id: NotRequired[str]    # derived from reply-chain root; scopes LLM history
    is_flat_thread: NotRequired[bool]  # True when the message is not a reply; flat mentions read recent chat context, not thread history
    worker_output: NotRequired[str | None] # raw facts gathered by the worker
    worker_tools_used: NotRequired[bool]   # True when the worker actually ran at least one tool (mechanical ToolMessage scan)
    search_notification_msg: NotRequired[Any]  # Telegram Message sent as search indicator
    response_messages: NotRequired[list]  # assembled LangChain messages forwarded to LanguageCorrectionNode
    humor_reply_to_msg_id: NotRequired[int | None]  # validated joke anchor; None sends the joke un-anchored
    is_bot_insult: NotRequired[bool]  # True when the filter classified the message as an insult aimed at the bot
    wind_down: NotRequired[bool]   # True when the engagement gate wants a short conversation-closing reply instead of a full one
    youtube_short_url: NotRequired[str | None]      # canonical Shorts URL, set by Router
    youtube_short_content: NotRequired[str | None]  # labelled transcript/frames/comments block, set by Ingester
    filter_verdict: NotRequired[str]   # "MEANINGFUL" | "MEANINGLESS" | "BANTER" | "BOT_INSULT" | "SHORTS", set by the filter node
    engagement_tier: NotRequired[int]  # wind-down tier charged by the engagement gate, set by the filter node
    drop_reason: NotRequired[str]      # why the pipeline ended without a reply, for the canonical log line
    media_is_real_person: NotRequired[bool | None]  # vision classification for photo/video_note/video, set by Ingester; None = text/voice/unclassified
