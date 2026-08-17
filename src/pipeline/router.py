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
  - The message is a direct reply to a bot message (text or any media type)
    — except a reply to the bot's own link-repost message (see
    src.events.link_repost), which needs a mention or a genuine
    question/request — reposted third-party content invites chat among
    members discussing it, not necessarily talk to the bot.
  - The text mentions the bot by word («бот» / "bot") without addressing it —
    routed with response_trigger="insult_check"; the filter node replies only
    if it confirms the message insults the bot.
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
    dedup window as Shorts (see ``src.pipeline.social_links``), but the
    daily cap is now per-handler — only Instagram has one (its anonymous
    fetch needs throttling against Instagram's anti-bot gate), and hitting
    it gets a canned reply instead of silence (``__daily_cap_reply``).
"""

import random
import re
from typing import Any

from src import log
from src.pipeline import shorts, social_links
from src.pipeline.state import BotState, IncomingMessage
from src.store import unified_messages

logger = log.get_logger(__name__)

# Only Instagram's handler still has a daily_cap (see social_links/__init__.py) —
# these acknowledge the limit instead of leaving a posted link answered with
# silence, same tone as the honest-failure pools in filter_node.py.
INSTAGRAM_REEL_DAILY_CAP_REPLIES = [
    "На сегодня лимит рилсов исчерпан — эту ссылку не разбираю.",
    "Дневной лимит рилсов выбран. Дальше сами, я на паузе до завтра.",
    "Рилсы на сегодня закончились — лимит. Возвращайтесь завтра.",
]


def is_mentioned(text: str, bot_username: str) -> bool:
    """Check whether text @mentions the bot on a word boundary.

    Args:
        text: Message text or caption to search.
        bot_username: Bot username, with or without the leading ``@``.

    Returns:
        ``True`` when ``@username`` appears as a whole word
        (case-insensitive — URLs and longer words merely containing the
        username do not count).
    """
    mention_pattern = rf"@{re.escape(bot_username.lstrip('@'))}\b"
    return bool(re.search(mention_pattern, text, re.IGNORECASE))


def is_reply_to_bot(telegram_message: Any, bot_id: int) -> bool:
    """Check whether a message directly replies to one of the bot's own messages.

    Args:
        telegram_message: The ``telegram.Message`` (or compatible) object.
        bot_id: Numeric Telegram id of the bot account.

    Returns:
        ``True`` when ``telegram_message.reply_to_message`` was sent by the bot.
    """
    reply = getattr(telegram_message, "reply_to_message", None)
    if not reply:
        return False
    sender = getattr(reply, "from_user", None)
    return sender is not None and sender.id == bot_id


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
    return is_mentioned(text, bot_username) or is_reply_to_bot(telegram_message, bot_id)

# Matches the word «бот» (in common Russian declensions) or "bot" as a whole
# word — a cheap precondition for the LLM insult check; deliberately excludes
# lookalikes such as «работа» or «ботан» via word boundaries and an explicit
# suffix list.
BOT_WORD_RE = re.compile(
    r"\b(?:бот(?:а|у|ом|е|ы|ов|ам|ами|ах)?|bot)\b",
    re.IGNORECASE,
)

# Leading interrogatives that mark a message as a real question even without
# a question mark — Russian and English. Relocated from filter_node.py:
# looks_like_request is now also used by MessageRouter's addressing gate for
# replies to link-repost messages (see __decide), so it needs to live
# somewhere both router.py and filter_node.py (which imports FROM router.py)
# can reach without a circular import.
QUESTION_WORDS = frozenset({
    "что", "чё", "че", "чо", "как", "почему", "зачем", "кто", "кого", "кому",
    "где", "когда", "куда", "откуда", "сколько", "какой", "какая", "какое",
    "какие", "каким", "чем", "чей", "чья", "чьё",
    "what", "how", "why", "who", "where", "when", "which", "whose",
})

# Leading imperative request verbs — a short command addressed to the bot is
# never meaningless, even when the classifier errs (e.g. a reply to a photo
# it could not see, like «переведи» under an unenriched meme).
REQUEST_WORDS = frozenset({
    "переведи", "переведите", "расскажи", "расскажите", "скажи", "скажите",
    "подскажи", "подскажите", "назови", "назовите",
    "покажи", "покажите", "напиши", "напишите", "объясни", "объясните",
    "поясни", "поясните", "сделай", "сделайте", "найди", "найдите",
    "проверь", "проверьте", "посчитай", "придумай", "кинь", "скинь",
    "дай", "давай", "помоги", "помогите",
    "поищи", "поищите", "загугли", "загуглите", "погугли", "погуглите",
    "гугли", "нагугли", "узнай", "узнайте",
    "translate", "tell", "show", "write", "make", "find", "check", "explain",
    "say", "give", "help", "search", "google", "lookup",
})

# A message with more than this many non-laughter word tokens is treated as
# substantive: every MEANINGLESS category is a SHORT reaction (laughter, «ок»,
# «бля», emoji, «хз»), so a longer message is essentially never meaningless.
SUBSTANTIVE_WORD_COUNT = 6

# Tokens that are pure laughter — skipped when looking for the leading word,
# so «ахаха что за бред» still reads as a question.
LAUGHTER_RE = re.compile(r"^(?:[хаеоы]+|[ha]+|l[ol]+|лол|кек|rofl|lmao)$", re.IGNORECASE)

# @handles are dropped before the leading-word analysis: an addressed message
# usually opens with «@bot …», and the handle would otherwise take the
# leading-word slot («@bot что это» reading as «bot») and inflate the word
# count. No handle is ever an interrogative or an imperative.
MENTION_RE = re.compile(r"@\w+")


def looks_like_request(text: str) -> bool:
    """Cheap deterministic check that a message is a question or imperative request.

    Used to override a MEANINGLESS verdict in the filter node (a question or
    request addressed to the bot always deserves a reply, however short it
    is — even when the classifier erred because the quoted content was
    opaque to it), and by MessageRouter's addressing gate for replies to
    link-repost messages (see __decide) — a bare reply to reposted content
    needs to look like a genuine request to count as addressing the bot.

    Every MEANINGLESS category is a SHORT reaction (laughter, «ок», «бля»,
    emoji, «хз»), so a message with more than ``SUBSTANTIVE_WORD_COUNT``
    non-laughter word tokens is treated as substantive regardless of its
    leading word — this catches long requests like «поищи в интернете, когда…»
    that a weak classifier mislabels and that no leading-word check would save.

    The text arrives as the user typed it, so an addressed message still
    carries its «@bot» handle; handles are stripped before tokenizing, or
    every @mentioned question would be judged on the bot's own username.

    Args:
        text: Raw message text.

    Returns:
        True when the text contains a question mark, has more than
        ``SUBSTANTIVE_WORD_COUNT`` non-laughter words, or its first
        non-laughter word is an interrogative from ``QUESTION_WORDS`` or an
        imperative from ``REQUEST_WORDS`` — all judged with @handles removed.
    """
    if "?" in text:
        return True
    without_mentions = MENTION_RE.sub(" ", text.lower())
    words = [word for word in re.findall(r"\w+", without_mentions) if not LAUGHTER_RE.fullmatch(word)]
    if len(words) > SUBSTANTIVE_WORD_COUNT:
        return True
    return bool(words) and words[0] in (QUESTION_WORDS | REQUEST_WORDS)


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
            shorts_update = self.__detect_shorts(msg)
            if shorts_update is not None:
                return shorts_update
            social_link_update = self.__detect_social_link(msg)
            if social_link_update is not None:
                return social_link_update

        should_respond, response_trigger = await self.__decide(msg, message)

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
            when there is no Shorts link or a gate rejected it. The update
            also carries ``link_message_is_bare``, which decides whether the
            events layer may delete the original message.
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
            "link_message_is_bare": social_links.is_bare_link_message(
                msg["raw_text"], shorts.SHORTS_URL_RE
            ),
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
            The update also carries ``link_message_is_bare``, which decides
            whether the events layer may delete the original message, and
            ``social_link_cap_remaining`` (None for an uncapped handler).
        """
        if handler.dedup_gate.seen((msg["chat_id"], handler.name, item_id)):
            logger.info(
                "%s repost in chat %s (%s) — skipping summary",
                handler.name, msg["chat_id"], item_id,
            )
            return None
        cap_remaining = None
        if handler.daily_cap is not None:
            used = handler.daily_cap_gate.hit(msg["chat_id"])
            if used > handler.daily_cap:
                return self.__daily_cap_reply(handler, msg["chat_id"], used)
            cap_remaining = handler.daily_cap - used
        return {
            "should_respond": True,
            "response_trigger": "social_link",
            "social_link_handler": handler.name,
            "social_link_url": canonical_url,
            "social_link_cap_remaining": cap_remaining,
            "link_message_is_bare": social_links.is_bare_link_message(
                msg["raw_text"], handler.pattern
            ),
        }

    def __daily_cap_reply(self, handler, chat_id: int, used: int) -> dict:
        """Log the cap rejection and answer with a canned limit notice.

        Only Instagram's handler carries a daily_cap today, so
        ``INSTAGRAM_REEL_DAILY_CAP_REPLIES`` is the only pool needed — a
        capped ``handler`` other than Instagram would need its own pool
        added here before this can serve it honestly.

        Args:
            handler: The matched ``LinkHandler`` whose cap was exceeded.
            chat_id: Telegram chat id the link was posted in.
            used: Hits recorded for this chat in the current window.

        Returns:
            State update dict: no fetch, but a visible reply instead of
            the original silent drop.
        """
        logger.warning(
            "%s daily cap reached for chat %s (%d/%d) — skipping summary",
            handler.name, chat_id, used, handler.daily_cap,
        )
        return {
            "should_respond": False,
            "response": random.choice(INSTAGRAM_REEL_DAILY_CAP_REPLIES),
        }

    async def __is_link_repost_reply(self, chat_id: int, reply: Any) -> bool:
        """Check whether a replied-to bot message is a link-repost row.

        Reuses the ``link_material`` column (set only for the bot's own
        link-repost messages — see src.events.messages.deliver_and_record)
        as the marker: any other bot message (jokes, roasts, voice replies)
        has it ``NULL``.

        Args:
            chat_id: Chat the reply belongs to.
            reply: The ``telegram.Message`` being replied to (already
                confirmed sent by the bot).

        Returns:
            ``True`` when the stored row carries persisted link material. A
            missing or purged row degrades to ``False`` — never gate more
            aggressively on missing data than on a confirmed non-link-repost
            row.
        """
        row = await unified_messages.get_by_id(chat_id=chat_id, message_id=reply.message_id)
        return bool(row and row.get("link_material"))

    async def __decide(self, msg: IncomingMessage, telegram_message: Any) -> tuple[bool, str]:
        """Pick the routing decision for one incoming message.

        A reply to the bot's link-repost message (see
        src.events.link_repost) needs a mention or request-like phrasing to
        count as addressing the bot — replying to reposted third-party
        content is often chat among members discussing the video, not
        talking to the bot, unlike a reply to the bot's own conversational
        output. Every other bot message keeps the blanket rule: any reply
        counts as addressing it. The router runs before transcription, so a
        non-text reply has no content for looks_like_request to judge —
        only an explicit mention can satisfy the gate for those.

        Args:
            msg: Normalised incoming-message dict from the pipeline state.
            telegram_message: The underlying ``telegram.Message`` object.

        Returns:
            Tuple of ``(should_respond, response_trigger)``.
        """
        if msg["is_forwarded"]:
            return False, "random"

        media_type = msg["media_type"]
        # Matches is_explicitly_addressed's original derivation exactly: text
        # OR caption, read from the Telegram object itself, not msg["raw_text"]
        # — a photo/voice message's @mention lives in its caption, and
        # msg["raw_text"] is not guaranteed to carry it (it does in
        # production, via build_pipeline_state, but nothing here should rely
        # on that indirection when the source object is right here).
        text = (
            getattr(telegram_message, "text", None)
            or getattr(telegram_message, "caption", None)
            or ""
        )
        mentioned = is_mentioned(text, self.__bot_username)
        reply_to_bot = is_reply_to_bot(telegram_message, self.__bot_id)
        addressed = mentioned or reply_to_bot
        if reply_to_bot and not mentioned:
            reply = telegram_message.reply_to_message
            if await self.__is_link_repost_reply(msg["chat_id"], reply):
                addressed = looks_like_request(text)

        if media_type == "text":
            if addressed:
                return True, "explicit"
            if BOT_WORD_RE.search(msg["raw_text"] or ""):
                return True, "insult_check"
            return False, "random"

        if media_type in ("voice", "video_note", "video", "photo"):
            if addressed:
                return True, "explicit"
            return False, "random"

        return False, "random"
