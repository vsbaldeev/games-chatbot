"""
MessageRouter — first node in the LangGraph pipeline.

Responsibilities:
  1. Persist every incoming message to unified_messages (as text or placeholder).
  2. Decide whether the bot should respond (sets should_respond).

Passive memory extraction for long plain-text messages that won't get a
response is not fired here — route_after_router (graph.py) sends those to the
memory_writer node instead, so extraction has exactly one call site.

Respond when:
  - The bot is @mentioned in the text / caption as a whole word
    (word-boundary match — URLs or longer words containing the username
    do not count).
  - The message is a direct reply to a bot message (text or any media type).
  - The text mentions the bot by word («бот» / "bot") without addressing it —
    routed with response_trigger="insult_check"; the filter node replies only
    if it confirms the message insults the bot.
  - Random chance fires for voice/video_note/video/photo (10%).
  - A text message contains a YouTube Shorts link — routed with
    response_trigger="youtube_short" so the pipeline summarizes the video.
    This check runs before the forwarded-message guard (forwarding is the
    dominant way links arrive, and «tell me what this video is» does not
    put words in the sender's mouth), but is gated by a per-chat repost
    dedup window and a daily summary cap (see ``src.pipeline.shorts``).
  - A text message contains an Instagram Reel or long-form YouTube video
    link (checked after Shorts, which keeps top priority) —
    routed with response_trigger="social_link" so the pipeline fetches a
    lightweight metadata-only summary (title/caption/selftext + top
    comments, no transcript or vision). The first handler in priority
    order whose regex matches wins the whole message; same per-item repost
    dedup window and daily summary cap pattern as Shorts (see
    ``src.pipeline.social_links``).
"""

import random
import re
from typing import Any

from src import log
from src.pipeline import humor_gate, shorts, social_links
from src.pipeline.state import BotState, IncomingMessage
from src.store import unified_messages
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

MEDIA_RESPONSE_CHANCE = 0.10

# Albums (shared media_group_id) roll the random-response dice once for the
# whole group instead of once per item. Albums arrive within seconds; the
# window only needs to outlive the slowest album delivery.
ALBUM_GATE_WINDOW_SECONDS = 5 * 60
album_gate = TtlGate(ALBUM_GATE_WINDOW_SECONDS)


def is_explicitly_addressed(telegram_message: Any, bot_username: str, bot_id: int) -> bool:
    """Check whether a Telegram message explicitly addresses the bot.

    A message is explicitly addressed when its text or caption mentions the
    bot as ``@username`` on a word boundary (case-insensitive — URLs and
    longer words merely containing the username do not count), or when it
    replies to one of the bot's own messages.

    Args:
        telegram_message: The ``telegram.Message`` (or compatible) object.
        bot_username: Bot username, with or without the leading ``@``.
        bot_id: Numeric Telegram id of the bot account.

    Returns:
        ``True`` when the message mentions the bot or replies to it.
    """
    text = (
        getattr(telegram_message, "text", None)
        or getattr(telegram_message, "caption", None)
        or ""
    )
    mention_pattern = rf"@{re.escape(bot_username.lstrip('@'))}\b"
    if re.search(mention_pattern, text, re.IGNORECASE):
        return True
    reply = getattr(telegram_message, "reply_to_message", None)
    if not reply:
        return False
    sender = getattr(reply, "from_user", None)
    return sender is not None and sender.id == bot_id

# Matches the word «бот» (in common Russian declensions) or "bot" as a whole
# word — a cheap precondition for the LLM insult check; deliberately excludes
# lookalikes such as «работа» or «ботан» via word boundaries and an explicit
# suffix list.
BOT_WORD_RE = re.compile(
    r"\b(?:бот(?:а|у|ом|е|ы|ов|ам|ами|ах)?|bot)\b",
    re.IGNORECASE,
)


class MessageRouter:
    """Determines whether the bot should respond and writes the message to the store."""

    def __init__(self, bot_username: str, bot_id: int) -> None:
        self.__bot_username = bot_username.lower()
        self.__bot_id = bot_id

    async def __call__(self, state: BotState) -> dict:
        msg: IncomingMessage = state["incoming"]
        update = msg["update"]
        message = update.message

        await self.__store_message(msg)

        if msg["media_type"] == "text":
            humor_gate.observe(msg["chat_id"])
            shorts_update = self.__detect_shorts(msg)
            if shorts_update is not None:
                return shorts_update
            social_link_update = self.__detect_social_link(msg)
            if social_link_update is not None:
                return social_link_update

        should_respond, response_trigger = self.__decide(msg, message)

        return {"should_respond": should_respond, "response_trigger": response_trigger}

    async def __store_message(self, msg: IncomingMessage) -> None:
        media_type = msg["media_type"]

        if media_type == "text":
            content = msg["raw_text"] or ""
        elif media_type == "voice":
            content = unified_messages.VOICE_PLACEHOLDER
        elif media_type == "video_note":
            content = unified_messages.VIDEO_NOTE_PLACEHOLDER
        elif media_type == "video":
            content = unified_messages.VIDEO_PLACEHOLDER
        elif media_type == "photo":
            content = unified_messages.format_photo_content(msg["raw_text"])
        elif media_type == "sticker":
            content = unified_messages.STICKER_PLACEHOLDER
        elif media_type == "animation":
            content = unified_messages.ANIMATION_PLACEHOLDER
        elif media_type == "audio":
            content = unified_messages.AUDIO_PLACEHOLDER
        else:
            content = ""

        try:
            await unified_messages.insert(
                chat_id=msg["chat_id"],
                message_id=msg["message_id"],
                user_id=msg["user_id"],
                username=msg["username"],
                content=content,
                media_type=media_type,
                reply_to_msg_id=msg["reply_to_msg_id"],
                file_id=msg["file_id"],
                media_group_id=msg.get("media_group_id"),
                is_forwarded=bool(msg.get("is_forwarded")),
            )
        except Exception as err:
            logger.warning("Failed to store message %s: %s", msg["message_id"], err)

    def __detect_shorts(self, msg: IncomingMessage) -> dict | None:
        """Route a YouTube Shorts link to the summary pipeline, if gates allow.

        Runs before the normal routing decision (and therefore before the
        forwarded-message guard). A gated link — repost within the dedup
        window or a chat over its daily summary cap — returns ``None`` and
        falls through to the normal decision, costing zero downloads and
        zero LLM tokens.

        Args:
            msg: Normalised incoming-message dict from the pipeline state.

        Returns:
            State update dict with the ``youtube_short`` trigger, or ``None``
            when there is no Shorts link or a gate rejected it.
        """
        video_id = shorts.extract_video_id(msg["raw_text"])
        if video_id is None:
            return None
        if shorts.dedup_gate.seen((msg["chat_id"], video_id)):
            logger.info("Shorts repost in chat %s (%s) — skipping summary", msg["chat_id"], video_id)
            return None
        if not shorts.under_daily_cap(msg["chat_id"]):
            return None
        return {
            "should_respond": True,
            "response_trigger": "youtube_short",
            "youtube_short_url": shorts.extract_shorts_url(msg["raw_text"]),
        }

    def __detect_social_link(self, msg: IncomingMessage) -> dict | None:
        """Route the first matching Instagram/YouTube link, if gates allow.

        Mirrors __detect_shorts: runs only when Shorts found nothing (Shorts
        keeps top priority) and tries social_links.HANDLERS in registry
        order. Only the first handler whose regex matches anything in the
        message is considered — every other link, same platform or
        different, is never looked at, even when that handler's own gate
        then rejects the match (see the design doc's multi-link precedence).

        Args:
            msg: Normalised incoming-message dict from the pipeline state.

        Returns:
            State update dict with the ``social_link`` trigger, or None when
            no handler matched or the matched handler's gate rejected it.
        """
        text = msg["raw_text"]
        for handler in social_links.HANDLERS:
            match = handler.extract(text)
            if match is None:
                continue
            item_id, canonical_url = match
            return self.__resolve_social_link_match(msg, handler, item_id, canonical_url)
        return None

    def __resolve_social_link_match(
        self, msg: IncomingMessage, handler, item_id: str, canonical_url: str,
    ) -> dict | None:
        """Gate-check the one matched handler and build its trigger update.

        Args:
            msg: Normalised incoming-message dict.
            handler: The matched ``LinkHandler``.
            item_id: Platform-specific id used for dedup gating.
            canonical_url: URL to fetch, passed through to the Ingester.

        Returns:
            State update dict on a gate pass, or None on a gate rejection.
        """
        if handler.dedup_gate.seen((msg["chat_id"], handler.name, item_id)):
            logger.info(
                "%s repost in chat %s (%s) — skipping summary",
                handler.name, msg["chat_id"], item_id,
            )
            return None
        used = handler.daily_cap_gate.hit(msg["chat_id"])
        if used > handler.daily_cap:
            logger.warning(
                "%s daily cap reached for chat %s (%d/%d) — skipping summary",
                handler.name, msg["chat_id"], used, handler.daily_cap,
            )
            return None
        return {
            "should_respond": True,
            "response_trigger": "social_link",
            "social_link_handler": handler.name,
            "social_link_url": canonical_url,
        }

    def __decide(self, msg: IncomingMessage, telegram_message: Any) -> tuple[bool, str]:
        """Pick the routing decision for one incoming message.

        Args:
            msg: Normalised incoming-message dict from the pipeline state.
            telegram_message: The underlying ``telegram.Message`` object.

        Returns:
            Tuple of ``(should_respond, response_trigger)``.
        """
        if msg["is_forwarded"]:
            return False, "random"

        media_type = msg["media_type"]
        addressed = is_explicitly_addressed(
            telegram_message, self.__bot_username, self.__bot_id
        )

        if media_type == "text":
            if addressed:
                return True, "explicit"
            if BOT_WORD_RE.search(msg["raw_text"] or ""):
                return True, "insult_check"
            return False, "random"

        if media_type in ("voice", "video_note", "video", "photo"):
            if addressed:
                return True, "explicit"
            media_group_id = msg.get("media_group_id")
            if media_group_id and album_gate.seen((msg["chat_id"], media_group_id)):
                return False, "random"
            return random.random() < MEDIA_RESPONSE_CHANCE, "random"

        return False, "random"
