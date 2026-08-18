"""
MeaninglessFilterNode — third node in the LangGraph pipeline.

Uses an LLM to classify the message (text or transcribed media) depending on
how it entered the pipeline:
  - Addressed messages (@mention / reply to the bot) are classified as
    MEANINGLESS (emoji reaction, no reply), BANTER (content-free jab that
    only keeps the exchange going), BOT_INSULT, PHOTO_REQUEST (the user asks
    for a photo of the bot itself — accepted at the full tier, the reply is
    an in-character «ща сфоткаю» ack and the events layer launches the
    background selfie generation; see ``src/life/selfie.py``) or MEANINGFUL
    (normal reply).
    When the message replies to an earlier stored
    message, that message is loaded and shown to the classifier as context —
    a short reaction like «ахаха что?» is meaningless in a vacuum but a real
    question when it quotes the bot's joke. A replied-to photo or sticker
    row still in placeholder form (e.g. the bot's own posted meme) is
    vision-enriched first via the shared ingester.enrich_media_row helper
    (cached to the store; sticker descriptions also cached per sticker
    identity), so «переведи» under a meme is classified against the image's
    actual content; when enrichment is impossible (no file_id, vision
    failure, animated sticker) or the row is another bare media placeholder
    ([voice], [animation]…), the placeholder is hidden and the reply is
    classified context-free rather than against a token the classifier
    cannot see.
    A MEANINGLESS or BANTER verdict on a text that looks like a question or
    an imperative request (question mark, leading interrogative, or a request
    verb like «переведи»/«расскажи», judged with @handles stripped) is
    overridden to MEANINGFUL: a question or request is never meaningless.
    The override is a free deterministic floor, not the primary defence:
    that is FILTER_MODEL itself, moved off the 8B model for dropping real
    questions and given a cross-provider fallback (make_filter_llm).
    A NOT_ADDRESSED verdict means the author replied to the bot but is
    talking about it to the chat (third person, venting, remarks to other
    members). It is honoured only when the router set ``broadcast_reply``
    — an un-mentioned reply to one of the bot's group-wide announcements
    — and resolves to full silence: no reply, no emoji, no attention-budget
    charge, because the bot was never addressed. Everywhere else the verdict
    is downgraded to MEANINGFUL, preserving this node's fail-open bias.
  - Overheard messages routed by the router's bot-word check
    (response_trigger="insult_check") are classified with the last few chat
    messages as context, so the model can tell this bot from game bots,
    other Telegram bots and people playing «как бот». A BOT_INSULT verdict
    acts only after a stronger model confirms it on the same input
    (disagreement or a confirmation error resolves to silence); everything
    else is dropped silently — no emoji reaction, because the bot was never
    addressed and reacting would be noise.
  - Media whose transcription/vision processing produced no text: explicitly
    addressed messages get an honest canned «не расслышал / не разглядел»
    reply (no LLM call); random-trigger media gets an emoji reaction and
    silence.
  - YouTube Shorts triggers (response_trigger="youtube_short") bypass the
    LLM classification entirely — a bare link would be classified
    MEANINGLESS and dropped, but the trigger is deterministic. A successful
    summary passes through; a failed one gets a canned «не смог посмотреть»
    reply when the sender explicitly addressed the bot, and full silence
    otherwise (no emoji reaction — the bot was never addressed).
  - Social-link triggers (response_trigger="social_link" — Instagram Reel,
    long-form YouTube video) bypass the LLM classification the
    same way as Shorts: a successful fetch passes through, a failed one
    gets the same canned-reply-if-addressed/full-silence-otherwise
    treatment.

Every verdict charges the sender's persistent attention budget (see
``engagement_gate``), and the post-charge wind-down tier decides the shape of
the reaction: a full reply, a short LLM-generated in-character brush-off
(``wind_down`` state flag), a bored emoji reaction, or silence — so any
sustained conversation fades out like a person losing interest. Counter-insults
that reply to the bot's own message never earn a fresh full comeback (that is
what fuels roast-battle loops).

Emoji reactions fire via asyncio.create_task and do not block the pipeline.
"""

import asyncio
import random
import re

import groq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from telegram import ReactionTypeEmoji

from src import config, log
from src.config.prompts import (
    BOT_INSULT_EXAMPLES,
    CRUDE_PRAISE_EXAMPLES,
    FILTER_SYSTEM,
    GROUP_PROFILE_COOLDOWN_REPLIES,
    GROUP_PROFILE_COOLDOWN_SECONDS,
    OVERHEARD_SYSTEM,
)
from src.pipeline import engagement_gate
from src.pipeline.ingester import enrich_media_row
from src.pipeline.memory_writer import MIN_PASSIVE_LENGTH, extract_and_save
from src.pipeline.router import is_explicitly_addressed, looks_like_request
from src.pipeline.state import BotState
from src.store import unified_messages
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

REACTION_POOL = ["👍", "❤", "🔥", "😁", "👀", "😎", "💯", "🤣", "⚡", "🫡", "🎉"]

# The classifiers answer with one label, so the completion budget only has to
# cover the longest of them plus the odd stray token.
FILTER_MAX_TOKENS = 10

# Bored acknowledgement for users the engagement gate has wound down to the
# emoji tier — must stay within Telegram's fixed set of allowed reaction emoji.
DISMISSIVE_REACTIONS = ["🥱", "😴", "🗿", "🤨"]

# Honest canned acknowledgements for explicitly addressed media whose
# transcription/vision processing produced nothing — deterministic, so the
# response model never improvises a reaction to content it never perceived.
TRANSCRIPTION_FAILED_REPLIES = [
    "Не расслышал ни слова. Перезапиши или скинь текстом.",
    "Войс не прожевался. Повтори?",
    "У меня уши отвалились — что там было?",
    "Звук до меня не доехал. Ещё раз?",
]

VISION_FAILED_REPLIES = [
    "Не разглядел, что там. Перезалей?",
    "Картинка до меня не доехала. Скинь ещё раз или расскажи словами.",
]

# Honest canned acknowledgements for an explicitly addressed Shorts link the
# bot failed to download or extract content from — same principle as the
# transcription-failure pools: never improvise a reaction to unseen content.
SHORTS_FAILED_REPLIES = [
    "Не смог посмотреть этот шортс — ютуб зажал. Смотрите сами.",
    "Ролик не открылся. Придётся рискнуть и смотреть вслепую.",
    "Ютуб мне этот шортс не отдал. Что я, впервые ему не угодил.",
    "Не дотянулся до ролика. Перекиньте другую ссылку или смотрите так.",
]

# Honest canned acknowledgements for an explicitly addressed Instagram/YouTube
# link the bot failed to fetch — same principle as SHORTS_FAILED_REPLIES.
SOCIAL_LINK_FAILED_REPLIES = [
    "Не смог посмотреть — сайт не отдал. Сами гляньте по ссылке.",
    "Ссылка не открылась. Придётся смотреть вслепую.",
    "Не дотянулся до контента. Перекиньте ещё раз или смотрите так.",
]

# Cap on how much replied-to text is fed to the classifier as context.
REPLIED_TO_CHAR_LIMIT = 500

# Any bare media placeholder ([photo], [sticker], [voice]…) is opaque to the
# classifier — showing the literal token invites a MEANINGLESS verdict on a
# reply that engages with content the classifier cannot see.
PLACEHOLDER_RE = re.compile(r"^\[\w+\]$")

# Deterministic floor under GROUP_PROFILE_REQUEST: FILTER_SYSTEM is a
# one-word classifier already carrying six labels, and this repo's own
# history (config/models.py) documents that model confusing labels under
# load. A GROUP_PROFILE_REQUEST verdict is trusted only when the raw text
# also names the whole chat — same relationship looks_like_request already
# has to MEANINGLESS/BANTER, a deterministic override on top of the
# classifier rather than a replacement for it.
GROUP_PROFILE_MARKER_RE = re.compile(
    r"\b(?:вс[её]м|всех|каждому|каждог[ао]|нам\s+вс[её]м|everyone|everybody|"
    r"each\s+of\s+(?:us|you)|all\s+of\s+(?:us|you))\b",
    re.IGNORECASE,
)

# Per-chat cooldown between accepted group-profile requests (see
# GROUP_PROFILE_COOLDOWN_SECONDS) — one run costs an LLM call plus a store
# query per chat member, well above an ordinary reply.
group_profile_cooldown_gate = TtlGate(GROUP_PROFILE_COOLDOWN_SECONDS)

# Overheard bot-word checks see the last few chat messages so the classifier
# can resolve which bot (or person playing «как бот») is being talked about.
OVERHEARD_CONTEXT_LIMIT = 5
OVERHEARD_CONTEXT_CHAR_LIMIT = 120

# Media types the unprompted random reaction skips when the vision classifier
# says the content is not a genuine photo/video of a real person (a meme,
# screenshot, art…). Explicit @mentions/replies are never gated — a member
# directly asking the bot to react to a meme still gets the roast.
RANDOM_MEDIA_MEME_GATE_TYPES = ("photo", "video_note")


def is_meme_random_trigger(state: BotState, media_type: str) -> bool:
    """True when a random-trigger photo/video note was classified as not a real person.

    Fails open: an unknown classification (``None`` — e.g. a vision-tag
    parsing hiccup) never suppresses a response.

    Args:
        state: Current pipeline state.
        media_type: The incoming message's media type.

    Returns:
        True only for a ``"random"`` trigger on a gated media type whose
        vision classification is explicitly False.
    """
    if state.get("response_trigger") != "random":
        return False
    if media_type not in RANDOM_MEDIA_MEME_GATE_TYPES:
        return False
    return state.get("media_is_real_person") is False


def replies_to_bot(msg: dict) -> bool:
    """Check whether the incoming message replies to one of the bot's own messages.

    Uses the ``replied_to_fallback`` synthesized from the Telegram update
    (always present for replies, no store lookup needed).

    Args:
        msg: IncomingMessage dict of the message being classified.

    Returns:
        True when the replied-to message was authored by the bot.
    """
    return (msg.get("replied_to_fallback") or {}).get("user_id") == config.BOT_ID


def replied_to_display_content(replied_to: dict) -> str:
    """Render a replied-to row's content for the classifier, hiding opaque placeholders.

    Args:
        replied_to: Stored or fallback row of the message being replied to.

    Returns:
        The content as-is for normal rows; the caption alone for photo rows
        still in placeholder form; empty string for bare media placeholders
        (``[photo]``, ``[sticker]``, ``[voice]``…) — the caller then falls
        back to context-free classification, where the reply is judged on
        its own merits instead of against a token the classifier cannot see.
    """
    content = (replied_to.get("content") or "").strip()
    if replied_to.get("media_type") == "photo" and unified_messages.needs_photo_description(content):
        return unified_messages.display_photo_content(content).strip()
    if PLACEHOLDER_RE.fullmatch(content):
        return ""
    return content


def build_filter_input(text: str, replied_to: dict | None) -> str:
    """Assemble the human message for the addressed-message classifier.

    Args:
        text: The user's message text.
        replied_to: Stored row of the message being replied to, or None.

    Returns:
        The bare text when there is no usable reply context (including a
        replied-to message that is only an opaque media placeholder);
        otherwise the replied-to message (author-labelled, truncated)
        followed by the user's reply, each marked so the classifier knows
        what to classify.
    """
    if replied_to is None:
        return text
    content = replied_to_display_content(replied_to)
    if not content:
        return text
    if len(content) > REPLIED_TO_CHAR_LIMIT:
        content = content[:REPLIED_TO_CHAR_LIMIT] + "…"
    if replied_to.get("user_id") == config.BOT_ID:
        author = "the bot"
    else:
        author = f"user @{replied_to.get('username')}"
    return (
        f"Message being replied to (from {author}):\n«{content}»\n\n"
        f"The user's reply (classify only this):\n«{text}»"
    )


def build_overheard_input(text: str, recent: list[dict]) -> str:
    """Assemble the human message for the overheard bot-word classifier.

    Args:
        text: The overheard message text under classification.
        recent: Recent chat rows (newest-first) providing referent context;
            may be empty, in which case the bare text is returned.

    Returns:
        The bare text when no context is available; otherwise the last few
        chat messages (oldest-first, truncated) followed by the message to
        classify, each section clearly labelled.
    """
    if not recent:
        return text
    lines = []
    for row in reversed(recent[:OVERHEARD_CONTEXT_LIMIT]):
        content = (row.get("content") or "").strip()
        if len(content) > OVERHEARD_CONTEXT_CHAR_LIMIT:
            content = content[:OVERHEARD_CONTEXT_CHAR_LIMIT] + "…"
        lines.append(f"@{row.get('username')}: {content}")
    context_block = "\n".join(lines)
    return (
        f"Recent chat context:\n{context_block}\n\n"
        f"Message to classify (only this):\n«{text}»"
    )


def make_filter_llm(model: str) -> Runnable:
    """Build a classification LLM with a cross-provider fallback.

    The filter decides whether a member who addressed the bot gets an answer
    at all, so a Groq quota exhaustion or outage must not turn into silence.
    The fallback runs the same weights through OpenRouter; without
    ``OPENROUTER_API_KEY`` the Groq client is returned bare and
    ``__classify`` degrades on its own (fail-open to MEANINGFUL).

    Any ``groq.APIError`` triggers the failover — rate limits, daily-quota
    429s a same-model retry can never recover from, connection errors and
    5xx alike. A request the fallback also rejects raises, and the caller's
    fail-open handles it.

    ``reasoning_effort="none"`` disables the primary's hidden reasoning —
    both FILTER_MODEL and INSULT_CONFIRM_MODEL are qwen/qwen3.6-27b, a
    reasoning model, and FILTER_MAX_TOKENS=10 leaves no room for a <think>
    block before the one-word label.

    Args:
        model: Groq model identifier for the primary client.

    Returns:
        A ``Runnable`` accepting a message list and returning one label.
    """
    primary_llm = ChatGroq(
        model=model,
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        max_tokens=FILTER_MAX_TOKENS,
        max_retries=0,
        reasoning_effort="none",
    )
    if not config.OPENROUTER_API_KEY:
        logger.warning("Filter: OPENROUTER_API_KEY unset — no fallback for %s", model)
        return primary_llm
    fallback_llm = ChatOpenAI(
        model=config.FILTER_FALLBACK_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=0.0,
        max_tokens=FILTER_MAX_TOKENS,
        max_retries=0,
    )
    return primary_llm.with_fallbacks([fallback_llm], exceptions_to_handle=(groq.APIError,))


class MeaninglessFilterNode:
    """LLM filter separating meaningless reactions, bot insults and real messages."""

    def __init__(self) -> None:
        """Build the classification and confirmation LLMs from configuration."""
        self.__llm = make_filter_llm(config.FILTER_MODEL)
        self.__confirm_llm = make_filter_llm(config.INSULT_CONFIRM_MODEL)

    async def __call__(self, state: BotState) -> dict:
        """Classify the incoming message and decide whether the bot replies.

        Args:
            state: Current pipeline state with the incoming message and the
                response trigger set by the router.

        Returns:
            State update dict; may set ``should_respond`` and ``is_bot_insult``.
        """
        if not state.get("should_respond"):
            return {}

        if state.get("response_trigger") == "youtube_short":
            return self.__handle_youtube_short(state)

        if state.get("response_trigger") == "social_link":
            return self.__handle_social_link(state)

        if state["incoming"]["media_type"] != "text":
            return await self.__handle_media(state)

        text = state["incoming"]["raw_text"] or ""
        if not text.strip():
            return {"should_respond": False}

        if state.get("response_trigger") == "insult_check":
            recent = await self.__fetch_recent_context(state["incoming"])
            overheard_input = build_overheard_input(text, recent)
            decision = await self.__classify(overheard_input, OVERHEARD_SYSTEM)
            return await self.__resolve_overheard(state, decision, overheard_input)

        decision = await self.__classify_addressed(state, text)
        if decision == "NOT_ADDRESSED":
            return self.__resolve_not_addressed(state)
        return await self.__resolve_with_budget(state, decision)

    async def __classify_addressed(self, state: BotState, text: str) -> str:
        """Classify an addressed message with reply context and a request override.

        Loads the replied-to message (if any) so the classifier sees what the
        user is reacting to — vision-enriching a photo or sticker row still
        in placeholder form first, so a reply to a meme is classified against
        the actual image content, not an opaque ``[photo]`` token. Then
        refuses to let a question or request be dropped: a MEANINGLESS or
        BANTER verdict on a text that looks like one is overridden to
        MEANINGFUL, keeping BOT_INSULT verdicts intact.

        A NOT_ADDRESSED verdict is honoured only when the router flagged the
        message as an un-mentioned reply to a broadcast; anywhere else it is
        downgraded to MEANINGFUL, so ordinary conversation can never be
        silenced by a classifier slip.

        Args:
            state: Current pipeline state.
            text: Raw message text.

        Returns:
            One of ``"BOT_INSULT"``, ``"BANTER"``, ``"MEANINGLESS"``,
            ``"PHOTO_REQUEST"``, ``"MEME_REQUEST"``, ``"GROUP_PROFILE_REQUEST"``,
            ``"NOT_ADDRESSED"`` or ``"MEANINGFUL"``.
        """
        replied_to = await self.__fetch_replied_to(state["incoming"])
        replied_to = await self.__enrich_replied_media(replied_to, state)
        decision = await self.__classify(build_filter_input(text, replied_to), FILTER_SYSTEM)
        decision = self.__downgrade_group_profile_request(state, text, decision)
        decision = self.__downgrade_not_addressed(state, decision)
        if decision in ("MEANINGLESS", "BANTER") and looks_like_request(text):
            logger.debug(
                "Filter: message %s is a question/request — overriding %s to MEANINGFUL",
                state["incoming"]["message_id"], decision,
            )
            return "MEANINGFUL"
        return decision

    def __downgrade_group_profile_request(self, state: BotState, text: str, decision: str) -> str:
        """Downgrade a GROUP_PROFILE_REQUEST verdict lacking a whole-chat marker.

        Args:
            state: Current pipeline state.
            text: Raw message text.
            decision: Verdict returned by ``__classify``.

        Returns:
            ``"MEANINGFUL"`` when ``decision`` is GROUP_PROFILE_REQUEST but the
            text names no whole-chat marker; ``decision`` unchanged otherwise.
        """
        if decision != "GROUP_PROFILE_REQUEST" or GROUP_PROFILE_MARKER_RE.search(text):
            return decision
        logger.debug(
            "Filter: message %s classified GROUP_PROFILE_REQUEST without a "
            "whole-chat marker — downgrading to MEANINGFUL",
            state["incoming"]["message_id"],
        )
        return "MEANINGFUL"

    def __downgrade_not_addressed(self, state: BotState, decision: str) -> str:
        """Downgrade a NOT_ADDRESSED verdict outside a broadcast reply.

        Ordinary conversation can never be silenced by this verdict: it is
        honoured only when the router flagged the message as an un-mentioned
        reply to a broadcast (see ``__resolve_not_addressed``).

        Args:
            state: Current pipeline state.
            decision: Verdict returned by ``__classify``.

        Returns:
            ``"MEANINGFUL"`` when ``decision`` is NOT_ADDRESSED but the
            message is not a broadcast reply; ``decision`` unchanged otherwise.
        """
        if decision != "NOT_ADDRESSED" or state.get("broadcast_reply"):
            return decision
        logger.debug(
            "Filter: message %s classified NOT_ADDRESSED outside a broadcast "
            "reply — downgrading to MEANINGFUL",
            state["incoming"]["message_id"],
        )
        return "MEANINGFUL"

    async def __enrich_replied_media(self, replied_to: dict | None, state: BotState) -> dict | None:
        """Vision-enrich an unenriched replied-to photo or sticker for the classifier.

        Delegates to ``ingester.enrich_media_row``, which no-ops for other
        media types, already-enriched rows, fallback rows without a file_id,
        and vision failures — and caches the result to the store, so the
        context builder reuses it instead of calling vision again.

        Args:
            replied_to: Row of the message being replied to, or None.
            state: Current pipeline state, providing the bot instance.

        Returns:
            The (possibly enriched) row, or None when there was no reply.
        """
        if replied_to is None:
            return None
        bot = state["context_types"].bot
        return await enrich_media_row(replied_to, state["incoming"]["chat_id"], bot)

    async def __fetch_replied_to(self, msg: dict) -> dict | None:
        """Load the message this one replies to: stored row first, update fallback second.

        Args:
            msg: IncomingMessage dict of the message being classified.

        Returns:
            The stored row of the replied-to message; when the row is missing
            or the lookup fails, the ``replied_to_fallback`` synthesized from
            the Telegram update; None when the message is not a reply.
        """
        reply_to_msg_id = msg["reply_to_msg_id"]
        if reply_to_msg_id is None:
            return None
        try:
            row = await unified_messages.get_by_id(
                chat_id=msg["chat_id"], message_id=reply_to_msg_id
            )
        except Exception as err:
            logger.warning(
                "Filter: failed to load replied-to message %s: %s", reply_to_msg_id, err
            )
            row = None
        return row or msg.get("replied_to_fallback")

    def __handle_youtube_short(self, state: BotState) -> dict:
        """Pass a summarized Shorts link through; degrade honestly on failure.

        The trigger is deterministic (a link was posted), so no LLM
        classification runs. When ingestion produced no content, an
        explicitly addressed sender gets a canned failure reply; an
        unaddressed link is dropped in full silence — no emoji reaction,
        because the bot was never addressed and reacting would be noise.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict.
        """
        if state.get("youtube_short_content"):
            return {"filter_verdict": "SHORTS"}
        message_id = state["incoming"]["message_id"]
        telegram_message = state["incoming"]["update"].message
        if telegram_message is not None and is_explicitly_addressed(
            telegram_message, config.BOT_USERNAME, config.BOT_ID
        ):
            logger.info(
                "Filter: no Shorts content for message %s, explicit trigger — canned failure reply",
                message_id,
            )
            return {"should_respond": False, "response": random.choice(SHORTS_FAILED_REPLIES)}
        logger.info("Filter: no Shorts content for message %s, skipping silently", message_id)
        return {"should_respond": False, "drop_reason": "shorts_failed"}

    def __handle_social_link(self, state: BotState) -> dict:
        """Pass a fetched Instagram/YouTube summary through; degrade honestly on failure.

        Mirrors __handle_youtube_short: the trigger is deterministic (a link
        was posted), so no LLM classification runs. Kept as its own method
        rather than merged with __handle_youtube_short — the two triggers
        are handled by entirely separate code paths by design (see the
        design doc's "Why Shorts stays separate" section), and merging them
        here would mean editing __handle_youtube_short's already-shipped
        body for a refactor this feature does not require.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict.
        """
        if state.get("social_link_content"):
            return {"filter_verdict": "SOCIAL_LINK"}
        message_id = state["incoming"]["message_id"]
        telegram_message = state["incoming"]["update"].message
        if telegram_message is not None and is_explicitly_addressed(
            telegram_message, config.BOT_USERNAME, config.BOT_ID
        ):
            logger.info(
                "Filter: no social-link content for message %s, explicit trigger — canned failure reply",
                message_id,
            )
            return {"should_respond": False, "response": random.choice(SOCIAL_LINK_FAILED_REPLIES)}
        logger.info("Filter: no social-link content for message %s, skipping silently", message_id)
        return {"should_respond": False, "drop_reason": "social_link_failed"}

    async def __handle_media(self, state: BotState) -> dict:
        """Pass media through when transcribed; degrade honestly when not.

        An explicitly addressed media message whose processing produced no
        text gets a canned «не расслышал / не разглядел» reply instead of a
        pass-through — generating a reaction from nothing is guaranteed
        hallucination. Unaddressed (random-trigger) media keeps the silent
        emoji-reaction path. A random-trigger photo/video note the vision
        classifier flagged as not a real person (a meme) is dropped the same
        way — silently, as if the random roll had simply missed.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict.
        """
        media_type = state["incoming"]["media_type"]
        text = state["incoming"]["processed_text"] or ""
        if text.strip():
            return await self.__handle_transcribed_media(state, media_type)
        if state.get("response_trigger") == "explicit":
            logger.info(
                "Filter: no transcription for %s message %s, explicit trigger — canned failure reply",
                media_type,
                state["incoming"]["message_id"],
            )
            pool = VISION_FAILED_REPLIES if media_type == "photo" else TRANSCRIPTION_FAILED_REPLIES
            return {"should_respond": False, "response": random.choice(pool)}
        logger.info(
            "Filter: no transcription for %s message %s, skipping",
            media_type,
            state["incoming"]["message_id"],
        )
        asyncio.create_task(self.__send_reaction(state))
        return {"should_respond": False, "drop_reason": "no_transcription"}

    async def __handle_transcribed_media(self, state: BotState, media_type: str) -> dict:
        """Pass a successfully transcribed/described media message through.

        A random-trigger photo/video note the vision classifier flagged as
        not a real person (a meme) is dropped instead — silently, as if the
        random roll had simply missed. Explicitly addressed media charges the
        sender's attention budget like a meaningful text turn, so switching
        to voice notes does not evade the wind-down; random-trigger media is
        bot-initiated and stays uncharged.

        Args:
            state: Current pipeline state.
            media_type: The incoming message's media type.

        Returns:
            State update dict.
        """
        if is_meme_random_trigger(state, media_type):
            logger.debug(
                "Filter: random-trigger %s message %s looks like a meme — skipping",
                media_type, state["incoming"]["message_id"],
            )
            return {"should_respond": False, "drop_reason": "meme_skip"}
        if state.get("response_trigger") == "explicit":
            return await self.__resolve_with_budget(state, "MEANINGFUL")
        return {}

    async def __resolve_with_budget(self, state: BotState, classification: str) -> dict:
        """Charge the sender's attention budget and apply the wind-down tier.

        Every classification spends budget (hostility and banter weigh more
        than a meaningful turn), and the post-charge tier decides the shape
        of the reaction.

        Args:
            state: Current pipeline state.
            classification: Filter verdict, a key of
                ``engagement_gate.SIGNAL_WEIGHTS``.

        Returns:
            State update dict.
        """
        msg = state["incoming"]
        tier = await engagement_gate.register_signal(
            chat_id=msg["chat_id"], user_id=msg["user_id"], classification=classification,
        )
        update = self.__apply_tier(state, classification, tier)
        return {"filter_verdict": classification, "engagement_tier": tier, **update}

    def __resolve_not_addressed(self, state: BotState) -> dict:
        """Drop a message whose author was talking about the bot, not to it.

        No attention budget is charged and no emoji reaction is sent: the bot
        was never addressed, so both would be the bot inserting itself into a
        conversation between members — the same reasoning the overheard-drop
        path already follows.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict ending the run without a reply.
        """
        logger.info(
            "Filter: message %s replies to a broadcast without addressing the bot — silence",
            state["incoming"]["message_id"],
        )
        return {
            "should_respond": False,
            "filter_verdict": "NOT_ADDRESSED",
            "drop_reason": "not_addressed",
        }

    def __apply_tier(self, state: BotState, classification: str, tier: int) -> dict:
        """Map (classification, tier) onto a reply, an emoji reaction or silence.

        Args:
            state: Current pipeline state.
            classification: Filter verdict.
            tier: Wind-down tier returned by the engagement gate.

        Returns:
            State update dict.
        """
        message_id = state["incoming"]["message_id"]
        if tier == engagement_gate.SILENCE_TIER:
            logger.debug(
                "Filter: %s from wound-down user, message %s — silence", classification, message_id
            )
            return {"should_respond": False, "drop_reason": "wound_down"}
        if classification == "MEANINGLESS":
            pool = DISMISSIVE_REACTIONS if tier == engagement_gate.EMOJI_TIER else REACTION_POOL
            logger.debug("Filter: Dropping meaningless message %s", message_id)
            asyncio.create_task(self.__send_reaction(state, pool))
            return {"should_respond": False, "drop_reason": "meaningless"}
        replies_with_text = tier == engagement_gate.FULL_TIER or (
            tier == engagement_gate.BRUSH_OFF_TIER and classification != "BANTER"
        )
        if not replies_with_text:
            logger.debug(
                "Filter: %s at tier %s, message %s — emoji reaction only",
                classification, tier, message_id,
            )
            asyncio.create_task(self.__send_reaction(state, DISMISSIVE_REACTIONS))
            return {"should_respond": False, "drop_reason": "wind_down_reaction"}
        return self.__build_reply_flags(state, classification, tier)

    def __build_reply_flags(self, state: BotState, classification: str, tier: int) -> dict:
        """Assemble the state flags for tiers that answer with text.

        A BANTER verdict and any brush-off-tier reply set ``wind_down`` (one
        short conversation-closing phrase, worker skipped). A BOT_INSULT gets
        the full comeback only at the full tier and only when the insult is
        not a reply to the bot's own message — a mirrored counter-insult in a
        running thread would just fuel the loop. A PHOTO_REQUEST at the full
        tier sets ``photo_request`` (the events layer launches generation
        after the ack) plus a ``photo_in_flight`` peek covering both image
        flows (``selfie.image_generation_in_flight``); at the brush-off tier
        it falls into the ``wind_down`` refusal. MEME_REQUEST behaves the same
        way via ``meme_request``, whose reply is the image alone.
        GROUP_PROFILE_REQUEST behaves the same way via ``group_profile_request``,
        except a request inside the per-chat cooldown window returns a canned
        refusal with ``should_respond: False`` instead — skipping the LLM call
        entirely rather than winding down the user's own attention budget.

        Args:
            state: Current pipeline state.
            classification: Filter verdict.
            tier: Wind-down tier returned by the engagement gate.

        Returns:
            State update dict, normally with ``should_respond: True``.
        """
        update: dict = {"should_respond": True}
        if classification == "BOT_INSULT":
            update["is_bot_insult"] = True
            if tier != engagement_gate.FULL_TIER or replies_to_bot(state["incoming"]):
                update["wind_down"] = True
        elif classification == "PHOTO_REQUEST" and tier == engagement_gate.FULL_TIER:
            from src.life import selfie
            update["photo_request"] = True
            update["photo_in_flight"] = selfie.image_generation_in_flight()
        elif classification == "MEME_REQUEST" and tier == engagement_gate.FULL_TIER:
            update["meme_request"] = True
        elif classification == "GROUP_PROFILE_REQUEST" and tier == engagement_gate.FULL_TIER:
            chat_id = state["incoming"]["chat_id"]
            if group_profile_cooldown_gate.seen(chat_id):
                return {
                    "should_respond": False,
                    "response": random.choice(GROUP_PROFILE_COOLDOWN_REPLIES),
                }
            update["group_profile_request"] = True
        elif classification == "BANTER" or tier != engagement_gate.FULL_TIER:
            update["wind_down"] = True
        logger.debug(
            "Filter: %s at tier %s, message %s — replying%s",
            classification, tier, state["incoming"]["message_id"],
            " (wind-down)" if update.get("wind_down") else "",
        )
        return update

    async def __fetch_recent_context(self, msg: dict) -> list[dict]:
        """Load recent chat rows for the overheard classifier, failing soft.

        Args:
            msg: IncomingMessage dict of the message being classified.

        Returns:
            Recent rows (newest-first, current message excluded), or an empty
            list when the lookup fails — degrading to context-free
            classification.
        """
        try:
            recent = await unified_messages.get_recent(
                chat_id=msg["chat_id"], limit=OVERHEARD_CONTEXT_LIMIT + 1
            )
            return [row for row in recent if row["message_id"] != msg["message_id"]]
        except Exception as err:
            logger.warning("Filter: failed to load overheard context: %s", err)
            return []

    async def __confirm_insult(self, overheard_input: str) -> bool:
        """Second opinion from the stronger model before an overheard comeback.

        Fail-soft is inverted here: for an unaddressed aggressive action the
        safe failure is silence, so any confirmation error drops the insult.

        Args:
            overheard_input: The exact input the cheap classifier saw.

        Returns:
            True only when the confirmation model also says BOT_INSULT.
        """
        try:
            response = await self.__confirm_llm.ainvoke([
                SystemMessage(content=OVERHEARD_SYSTEM),
                HumanMessage(content=overheard_input),
            ])
            verdict = response.content.strip().upper()
            logger.debug("Filter: overheard insult confirmation verdict: %s", verdict)
            return "INSULT" in verdict
        except Exception as err:
            logger.warning("Insult confirmation failed — dropping overheard insult: %s", err)
            return False

    async def __resolve_overheard(
        self, state: BotState, decision: str, overheard_input: str
    ) -> dict:
        """Apply the verdict for an overheard message that mentions the bot by word.

        A BOT_INSULT verdict from the cheap classifier acts only after the
        stronger model confirms it on the same input; disagreement (or a
        confirmation error) resolves as OTHER. Anything else is dropped
        silently — no emoji reaction — and long texts get the passive fact
        extraction the router skips for insult-check candidates.

        Args:
            state: Current pipeline state.
            decision: Cheap classifier verdict.
            overheard_input: The classifier input, re-used for confirmation.

        Returns:
            State update dict.
        """
        msg = state["incoming"]
        if decision == "BOT_INSULT" and await self.__confirm_insult(overheard_input):
            return await self.__resolve_with_budget(state, "BOT_INSULT")
        text = msg["raw_text"] or ""
        if len(text.strip()) >= MIN_PASSIVE_LENGTH:
            asyncio.create_task(extract_and_save(
                chat_id=msg["chat_id"],
                user_id=msg["user_id"],
                username=msg["username"],
                user_message=text,
            ))
        return {"should_respond": False, "drop_reason": "overheard_dropped"}

    async def __classify(self, text: str, system_prompt: str) -> str:
        """Classify the message text with the filter LLM.

        Args:
            text: Raw message text, or the reply-context input assembled by
                ``build_filter_input`` for addressed replies.
            system_prompt: Classification prompt — ``FILTER_SYSTEM`` for messages
                addressed to the bot, ``OVERHEARD_SYSTEM`` for bot-word mentions.

        Returns:
            One of ``"BOT_INSULT"``, ``"BANTER"``, ``"MEANINGLESS"``,
            ``"PHOTO_REQUEST"``, ``"MEME_REQUEST"``, ``"GROUP_PROFILE_REQUEST"``,
            ``"NOT_ADDRESSED"`` or ``"MEANINGFUL"``. Fails open to
            ``"MEANINGFUL"`` on any LLM error.
        """
        try:
            response = await self.__llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=text),
            ])
            result = response.content.strip().upper()
            if "PHOTO" in result:
                return "PHOTO_REQUEST"
            if "MEME" in result:
                return "MEME_REQUEST"
            if "PROFILE" in result:
                return "GROUP_PROFILE_REQUEST"
            if "NOT_ADDRESSED" in result:
                return "NOT_ADDRESSED"
            if "INSULT" in result:
                return "BOT_INSULT"
            if "BANTER" in result:
                return "BANTER"
            return "MEANINGLESS" if "MEANINGLESS" in result else "MEANINGFUL"
        except Exception as err:
            logger.warning("Meaningless filter failed, failing open (MEANINGFUL): %s", err)
            return "MEANINGFUL"

    async def __send_reaction(self, state: BotState, pool: list[str] = REACTION_POOL) -> None:
        """React to the message with a random emoji from the given pool.

        Args:
            state: Current pipeline state.
            pool: Emoji to pick from; defaults to the friendly acknowledgement
                pool, with ``DISMISSIVE_REACTIONS`` used for insult barrages.
        """
        try:
            bot = state["context_types"].bot
            msg = state["incoming"]
            emoji = random.choice(pool)
            await bot.set_message_reaction(
                chat_id=msg["chat_id"],
                message_id=msg["message_id"],
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception as err:
            logger.warning("Reaction failed for message %s: %s", state["incoming"]["message_id"], err)
