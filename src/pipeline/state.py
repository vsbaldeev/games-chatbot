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
    replied_to: dict | None            # the specific message being replied to, for annotation; may carry link_material (see src.pipeline.router's addressing gate)
    reply_chain: list[dict]            # full reply chain from root to replied-to message, oldest-first
    asking_user_tag: dict | None       # {"tag", "reason"} weekly role of the message sender; None unless the message is about roles
    mentioned_tags: dict[str, dict]    # username → {"tag", "reason"} for members @mentioned in the question


class BotState(TypedDict):
    """Full mutable state passed between LangGraph nodes."""

    incoming: IncomingMessage
    should_respond: bool
    response_trigger: str          # "explicit" (@mention/reply), "insult_check" (bot-word mention), "random" (unprompted text reply), "youtube_short" (Shorts link) or "social_link" (Instagram/YouTube link)
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
    is_bot_insult: NotRequired[bool]  # True when the filter classified the message as an insult aimed at the bot
    wind_down: NotRequired[bool]   # True when the engagement gate wants a short conversation-closing reply instead of a full one
    youtube_short_url: NotRequired[str | None]      # canonical Shorts URL, set by Router
    youtube_short_content: NotRequired[str | None]  # labelled transcript/frames/comments block, set by Ingester
    youtube_short_video: NotRequired[bytes | None]  # downloaded video bytes, set by Ingester
    social_link_handler: NotRequired[str | None]    # matched handler name ("instagram_reel"/"youtube_video"), set by Router
    social_link_url: NotRequired[str | None]        # canonical URL, set by Router
    social_link_content: NotRequired[str | None]    # labelled content block, set by Ingester
    social_link_video: NotRequired[bytes | None]    # downloaded video bytes (Instagram only), set by Ingester
    social_link_cap_remaining: NotRequired[int | None]  # daily_cap - used for this chat, set by Router; None when the matched handler has no cap
    link_message_is_bare: NotRequired[bool]  # True when the triggering message was the link and nothing else; only then may the events layer delete it
    broadcast_reply: NotRequired[bool]  # True when the message replies to one of the bot's group-wide announcements (is_broadcast row) without @mentioning it; only then may the filter return NOT_ADDRESSED
    filter_verdict: NotRequired[str]   # "MEANINGFUL" | "MEANINGLESS" | "BANTER" | "BOT_INSULT" | "PHOTO_REQUEST" | "MEME_REQUEST" | "GROUP_PROFILE_REQUEST" | "NOT_ADDRESSED" | "SHORTS" | "SOCIAL_LINK", set by the filter node
    photo_request: NotRequired[bool]   # True when the filter accepted a photo request at the full tier; the events layer launches selfie generation after the ack is delivered
    meme_request: NotRequired[bool]    # True when the filter accepted a meme request at the full tier; the response is empty and the events layer sends the meme itself
    group_profile_request: NotRequired[bool]  # True when the filter accepted a group-profile request at the full tier; the response is empty and the events layer generates+sends the profile itself
    photo_in_flight: NotRequired[bool] # True when a selfie was already being generated at classification time; the response acks «уже фоткаю» and no second job is launched
    engagement_tier: NotRequired[int]  # wind-down tier charged by the engagement gate, set by the filter node
    drop_reason: NotRequired[str]      # why the pipeline ended without a reply, for the canonical log line
    media_is_real_person: NotRequired[bool | None]  # vision classification for photo/video_note/video, set by Ingester; None = text/voice/unclassified
    voice_low_confidence: NotRequired[bool]  # True when Whisper's mean segment confidence was low for a voice transcript; response_node softens its reaction accordingly
