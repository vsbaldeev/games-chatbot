"""
Shared state types for the LangGraph pipeline.

BotState flows through the parent graph (router → ingest → respond → memory_writer).
Each stage subgraph sees only its own schema: IngestState for the ingester and filter
nodes, RespondState for guard → context → worker → response. LangGraph hides undeclared
keys from a stage and silently drops undeclared keys a node returns, so a key a stage
touches must be declared on its schema (tests/unit/pipeline/test_state_scoping.py).
"""

from enum import StrEnum
from typing import Any

from typing_extensions import NotRequired, TypedDict


class ContentKind(StrEnum):
    """Why a bot message exists — one value per row of bot_message_content."""

    RESPONSE = "response"                          # free-form reply written by the response LLM
    LINK_SUMMARY = "link_summary"                  # Shorts / Instagram / TikTok / YouTube summary, nobody asked a question
    WIND_DOWN_REPLY = "wind_down_reply"            # LLM reply under the wind-down directive
    INSULT_REPLY = "insult_reply"                  # LLM reply under the BOT_INSULT directive
    PROMPT_INJECTION_REPLY = "prompt_injection_reply"  # canned reply after the guard flagged the message (may be a false positive)
    LIMIT_NOTICE = "limit_notice"                  # Groq daily-quota / rate-limit notice
    CONTEXT_LIMIT_NOTICE = "context_limit_notice"  # context-window overflow notice
    CAP_NOTICE = "cap_notice"                      # Instagram + TikTok shared daily cap, group-profile cooldown
    MEDIA_FAILURE_NOTICE = "media_failure_notice"  # could not transcribe / see / fetch
    ERROR = "error"                                # generic failure line


class Verdict(StrEnum):
    """What the filter stage decided about a message; the value is stored and logged as-is."""

    MEANINGFUL = "MEANINGFUL"                      # a real message: normal reply
    MEANINGLESS = "MEANINGLESS"                    # content-free reaction: emoji, no reply
    BANTER = "BANTER"                              # content-free jab: reply under the wind-down directive
    BOT_INSULT = "BOT_INSULT"                      # insult aimed at the bot
    MEME_REQUEST = "MEME_REQUEST"                  # «скинь мем»
    GROUP_PROFILE_REQUEST = "GROUP_PROFILE_REQUEST"  # judge every member against a theme
    NOT_ADDRESSED = "NOT_ADDRESSED"                # author is talking about the bot, not to it
    SHORTS = "SHORTS"                              # YouTube Shorts link with fetched content
    SOCIAL_LINK = "SOCIAL_LINK"                    # Instagram / TikTok / YouTube link with fetched content


class WorkerStatus(StrEnum):
    """How the worker stage ended; absent when it never finished."""

    OK = "ok"                              # worker ran
    SKIPPED = "skipped"                    # deliberately not used (insult, wind-down, meme, group profile, links)
    CONTEXT_OVERFLOW = "context_overflow"  # ContextLengthError absorbed, empty worker output
    FAILED = "failed"                      # unexpected exception absorbed, empty worker output


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


class MessageState(TypedDict):
    """Keys every stage reads: the message itself, who it is for and the reply it may produce."""

    incoming: IncomingMessage
    should_respond: bool
    response_trigger: str          # "explicit" (@mention/reply), "insult_check" (bot-word mention), "random" (unprompted text reply), "youtube_short" (Shorts link) or "social_link" (Instagram/TikTok/YouTube link)
    context_types: Any             # telegram.ext.ContextTypes instance for sending replies
    response: str | None
    content_kind: NotRequired[ContentKind]  # set only by nodes that put a canned text in `response` (guard, filter, router); LLM-written kinds are derived at delivery
    drop_reason: NotRequired[str]      # why the pipeline ended without a reply, for the canonical log line


class FilterOutcome(TypedDict):
    """What the ingest stage decided; the respond stage reads it to shape the reply."""

    filter_verdict: NotRequired[Verdict]  # set by the filter stage; see Verdict
    is_bot_insult: NotRequired[bool]  # True when the filter classified the message as an insult aimed at the bot
    wind_down: NotRequired[bool]   # True when the engagement gate wants a short conversation-closing reply instead of a full one
    meme_request: NotRequired[bool]    # True when the filter accepted a meme request at the full tier; the response is empty and the events layer sends the meme itself
    group_profile_request: NotRequired[bool]  # True when the filter accepted a group-profile request at the full tier; the response is empty and the events layer generates+sends the profile itself
    voice_low_confidence: NotRequired[bool]  # True when Whisper's mean segment confidence was low for a voice transcript; response_node softens its reaction accordingly


class IngestState(MessageState, FilterOutcome):
    """Everything the ingest stage (ingester + filter nodes) reads or writes."""

    youtube_short_url: NotRequired[str | None]      # canonical Shorts URL, set by Router
    youtube_short_content: NotRequired[str | None]  # labelled transcript/frames/comments block, set by Ingester
    youtube_short_video: NotRequired[bytes | None]  # downloaded video bytes, set by Ingester
    social_link_handler: NotRequired[str | None]    # matched handler name ("instagram_reel"/"tiktok_video"/"youtube_video"), set by Router
    social_link_url: NotRequired[str | None]        # canonical URL, set by Router
    social_link_content: NotRequired[str | None]    # labelled content block, set by Ingester
    social_link_video: NotRequired[bytes | None]    # downloaded video bytes (Instagram/TikTok only), set by Ingester
    broadcast_reply: NotRequired[bool]  # True when the message replies to one of the bot's group-wide announcements (is_broadcast row) without @mentioning it; only then may the filter return NOT_ADDRESSED
    engagement_tier: NotRequired[int]  # wind-down tier charged by the engagement gate, set by the filter node
    media_is_real_person: NotRequired[bool | None]  # vision classification for photo/video_note/video, set by Ingester; None = text/voice/unclassified


class RespondState(MessageState, FilterOutcome):
    """Everything the respond stage (guard → context → worker → response) reads or writes."""

    blocked: bool                  # True when Guard Node rejects the message
    context: AssembledContext | None
    thread_id: NotRequired[str]    # derived from reply-chain root; scopes LLM history
    is_flat_thread: NotRequired[bool]  # True when the message is not a reply; flat mentions read recent chat context, not thread history
    worker_output: NotRequired[str | None] # raw facts gathered by the worker
    worker_tools_used: NotRequired[bool]   # True when the worker actually ran at least one tool (mechanical ToolMessage scan)
    search_notification_msg: NotRequired[Any]  # Telegram Message sent as search indicator
    worker_status: NotRequired[WorkerStatus]  # set by WorkerNode on every run; absent when the worker never finished
    response_messages: NotRequired[list]  # assembled LangChain messages forwarded to LanguageCorrectionNode
    response_trace: NotRequired[dict | None]  # exact prompt/response of the response LLM call, set by ResponseNode/LanguageCorrectionNode; None when no response LLM call ran (meme/group-profile requests)


class BotState(IngestState, RespondState):
    """Full state of one pipeline run: both stage schemas plus keys only the router and the events layer touch."""

    social_link_cap_remaining: NotRequired[int | None]  # daily_cap - used for this chat, set by Router; None when the matched handler has no cap
    link_message_is_bare: NotRequired[bool]  # True when the triggering message was the link and nothing else; only then may the events layer delete it
