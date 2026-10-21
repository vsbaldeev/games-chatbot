"""
ClassifyNode — filter-stage node that classifies a text message with an LLM.

Classification depends on how the message entered the pipeline:
  - Addressed messages (@mention / reply to the bot) are classified as
    MEANINGLESS (emoji reaction, no reply), BANTER (content-free jab that
    only keeps the exchange going), BOT_INSULT or MEANINGFUL (normal reply).
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
    charge, because the bot was never addressed; a passive fact extraction
    still runs first, so substantive self-description in the message is not
    lost. Even inside a broadcast reply, a text that looks like a genuine
    question or request (the same deterministic floor MEANINGLESS/BANTER
    get from looks_like_request) is never silenced this way — NOT_ADDRESSED
    is the one verdict that produces total silence, so a classifier misfire
    on it is the costliest to leave unguarded. Everywhere else the verdict
    is downgraded to MEANINGFUL, preserving this node's fail-open bias.
  - Overheard messages routed by the router's bot-word check
    (response_trigger="insult_check") are classified with the last few chat
    messages as context, so the model can tell this bot from game bots,
    other Telegram bots and people playing «как бот». A BOT_INSULT verdict
    acts only after a stronger model confirms it on the same input
    (disagreement or a confirmation error resolves to silence); everything
    else is dropped silently — no emoji reaction, because the bot was never
    addressed and reacting would be noise.

The node only decides the verdict (``filter_verdict``) or ends the run; the
attention-budget charge and the reaction happen in ``budget.charge_budget``.
"""

import asyncio
import re

import groq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src import config, log
from src.config.prompts import FILTER_SYSTEM, OVERHEARD_SYSTEM
from src.pipeline.ingester import enrich_media_row
from src.pipeline.memory_writer import MIN_PASSIVE_LENGTH, extract_and_save
from src.pipeline.router import looks_like_request
from src.pipeline.state import BotState, Verdict
from src.store import unified_messages

logger = log.get_logger(__name__)

# The classifiers answer with one label, so the completion budget only has to
# cover the longest of them plus the odd stray token.
FILTER_MAX_TOKENS = 10

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

# Overheard bot-word checks see the last few chat messages so the classifier
# can resolve which bot (or person playing «как бот») is being talked about.
OVERHEARD_CONTEXT_LIMIT = 5
OVERHEARD_CONTEXT_CHAR_LIMIT = 120


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


def parse_label(content: str) -> Verdict:
    """Map the classifier's free-text answer onto one of the filter labels.

    Args:
        content: Raw model output, e.g. ``"meaningless - just laughter"``.

    Returns:
        One of ``"BOT_INSULT"``, ``"BANTER"``, ``"MEANINGLESS"``,
        ``"MEME_REQUEST"``, ``"GROUP_PROFILE_REQUEST"``, ``"NOT_ADDRESSED"``
        or ``"MEANINGFUL"`` (the default for anything unrecognised).
    """
    result = content.strip().upper()
    if "MEME" in result:
        return Verdict.MEME_REQUEST
    if "PROFILE" in result:
        return Verdict.GROUP_PROFILE_REQUEST
    if "NOT_ADDRESSED" in result:
        return Verdict.NOT_ADDRESSED
    if "INSULT" in result:
        return Verdict.BOT_INSULT
    if "BANTER" in result:
        return Verdict.BANTER
    return Verdict.MEANINGLESS if "MEANINGLESS" in result else Verdict.MEANINGFUL


def make_filter_llm(model: str) -> Runnable:
    """Build a classification LLM with a cross-provider fallback.

    The filter decides whether a member who addressed the bot gets an answer
    at all, so a Groq quota exhaustion or outage must not turn into silence.
    The fallback runs the same weights through OpenRouter; without
    ``OPENROUTER_API_KEY`` the Groq client is returned bare and
    ``ClassifyNode.__classify`` degrades on its own (fail-open to MEANINGFUL).

    Any ``groq.APIError`` triggers the failover — rate limits, daily-quota
    429s a same-model retry can never recover from, connection errors and
    5xx alike. A request the fallback also rejects raises, and the caller's
    fail-open handles it.

    ``reasoning_effort="none"`` disables the primary's hidden reasoning —
    both FILTER_MODEL and INSULT_CONFIRM_MODEL are qwen/qwen3.8-27b, a
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


class ClassifyNode:
    """LLM classifier for text messages: addressed replies and overheard bot-word mentions."""

    def __init__(self, llm: Runnable | None = None, confirm_llm: Runnable | None = None) -> None:
        """Build the classification and confirmation LLMs unless they are injected.

        Args:
            llm: Classifier LLM; built from ``config.FILTER_MODEL`` when None.
            confirm_llm: Overheard-insult confirmation LLM; built from
                ``config.INSULT_CONFIRM_MODEL`` when None.
        """
        self.__llm = llm if llm is not None else make_filter_llm(config.FILTER_MODEL)
        self.__confirm_llm = (
            confirm_llm if confirm_llm is not None else make_filter_llm(config.INSULT_CONFIRM_MODEL)
        )

    async def __call__(self, state: BotState) -> dict:
        """Classify the incoming text message.

        Args:
            state: Current pipeline state with a text message and the response
                trigger set by the router.

        Returns:
            ``{"filter_verdict": label}`` when the message goes on to the
            budget node; otherwise a drop update with ``should_respond=False``.
        """
        text = state["incoming"]["raw_text"] or ""
        if not text.strip():
            return {"should_respond": False}

        if state.get("response_trigger") == "insult_check":
            recent = await self.__fetch_recent_context(state["incoming"])
            overheard_input = build_overheard_input(text, recent)
            decision = await self.__classify(overheard_input, OVERHEARD_SYSTEM)
            return await self.__resolve_overheard(state, decision, overheard_input)

        decision = await self.__classify_addressed(state, text)
        if decision == Verdict.NOT_ADDRESSED:
            return self.__resolve_not_addressed(state)
        return {"filter_verdict": decision}

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
        message as an un-mentioned reply to a broadcast, and only when the
        text does not itself look like a question or request; anywhere else
        it is downgraded to MEANINGFUL, so ordinary conversation — or a real
        question a classifier misjudged — can never be silenced.

        Args:
            state: Current pipeline state.
            text: Raw message text.

        Returns:
            One of ``"BOT_INSULT"``, ``"BANTER"``, ``"MEANINGLESS"``,
            ``"MEME_REQUEST"``, ``"GROUP_PROFILE_REQUEST"``,
            ``"NOT_ADDRESSED"`` or ``"MEANINGFUL"``.
        """
        replied_to = await self.__fetch_replied_to(state["incoming"])
        replied_to = await self.__enrich_replied_media(replied_to, state)
        decision = await self.__classify(build_filter_input(text, replied_to), FILTER_SYSTEM)
        decision = self.__downgrade_group_profile_request(state, text, decision)
        decision = self.__downgrade_not_addressed(state, text, decision)
        if decision in (Verdict.MEANINGLESS, Verdict.BANTER) and looks_like_request(text):
            logger.debug(
                "Filter[classify]: message %s is a question/request — overriding %s to MEANINGFUL",
                state["incoming"]["message_id"], decision,
            )
            return Verdict.MEANINGFUL
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
        if decision != Verdict.GROUP_PROFILE_REQUEST or GROUP_PROFILE_MARKER_RE.search(text):
            return decision
        logger.debug(
            "Filter[classify]: message %s classified GROUP_PROFILE_REQUEST without a "
            "whole-chat marker — downgrading to MEANINGFUL",
            state["incoming"]["message_id"],
        )
        return Verdict.MEANINGFUL

    def __downgrade_not_addressed(self, state: BotState, text: str, decision: str) -> str:
        """Downgrade a NOT_ADDRESSED verdict outside a broadcast reply, or on a real request.

        Ordinary conversation can never be silenced by this verdict: it is
        honoured only when the router flagged the message as an un-mentioned
        reply to a broadcast (see ``__resolve_not_addressed``) — and even
        then only when the text does not look like a genuine question or
        imperative request. This mirrors the MEANINGLESS/BANTER override in
        ``__classify_addressed``, applied here as well because NOT_ADDRESSED
        is the only verdict that produces total silence, so a classifier
        misfire on it deserves the same deterministic floor.

        Args:
            state: Current pipeline state.
            text: Raw message text.
            decision: Verdict returned by ``__classify``.

        Returns:
            ``"MEANINGFUL"`` when ``decision`` is NOT_ADDRESSED but the
            message is not a broadcast reply, or looks like a question or
            request; ``decision`` unchanged otherwise.
        """
        if decision != Verdict.NOT_ADDRESSED:
            return decision
        if state.get("broadcast_reply") and not looks_like_request(text):
            return decision
        logger.debug(
            "Filter[classify]: message %s classified NOT_ADDRESSED — downgrading to "
            "MEANINGFUL (not a broadcast reply, or looks like a request)",
            state["incoming"]["message_id"],
        )
        return Verdict.MEANINGFUL

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
        except Exception as err:  # noqa: BLE001 — fail soft to the update's fallback row
            logger.warning(
                "Filter[classify]: failed to load replied-to message %s: %s", reply_to_msg_id, err
            )
            row = None
        return row or msg.get("replied_to_fallback")

    def __resolve_not_addressed(self, state: BotState) -> dict:
        """Drop a message whose author was talking about the bot, not to it.

        No attention budget is charged and no emoji reaction is sent: the bot
        was never addressed, so both would be the bot inserting itself into a
        conversation between members — the same reasoning the overheard-drop
        path already follows. Passive fact extraction still runs before the
        drop (see ``__extract_passive_facts``), so substantive
        self-description in the message is not silently lost.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict ending the run without a reply.
        """
        logger.info(
            "Filter[classify]: message %s replies to a broadcast without addressing the bot — silence",
            state["incoming"]["message_id"],
        )
        self.__extract_passive_facts(state["incoming"])
        return {
            "should_respond": False,
            "filter_verdict": Verdict.NOT_ADDRESSED,
            "drop_reason": "broadcast_not_addressed",
        }

    def __extract_passive_facts(self, msg: dict) -> None:
        """Fire-and-forget passive fact extraction before a message is dropped silently.

        Shared by the overheard-drop and NOT_ADDRESSED-drop paths, both of
        which end the run without a reply but must not lose substantive
        self-description the sender typed along the way.

        Args:
            msg: IncomingMessage dict of the message being dropped.
        """
        text = msg["raw_text"] or ""
        if len(text.strip()) >= MIN_PASSIVE_LENGTH:
            asyncio.create_task(extract_and_save(
                chat_id=msg["chat_id"],
                user_id=msg["user_id"],
                username=msg["username"],
                user_message=text,
            ))

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
        except Exception as err:  # noqa: BLE001 — fail soft to context-free classification
            logger.warning("Filter[classify]: failed to load overheard context: %s", err)
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
            logger.debug("Filter[classify]: overheard insult confirmation verdict: %s", verdict)
            return "INSULT" in verdict
        except Exception as err:  # noqa: BLE001 — any confirmation error resolves to silence
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
            ``{"filter_verdict": "BOT_INSULT"}`` for a confirmed insult;
            otherwise a drop update.
        """
        if decision == Verdict.BOT_INSULT and await self.__confirm_insult(overheard_input):
            return {"filter_verdict": Verdict.BOT_INSULT}
        self.__extract_passive_facts(state["incoming"])
        return {"should_respond": False, "drop_reason": "overheard_dropped"}

    async def __classify(self, text: str, system_prompt: str) -> str:
        """Classify the message text with the filter LLM.

        Args:
            text: Raw message text, or the reply-context input assembled by
                ``build_filter_input`` for addressed replies.
            system_prompt: Classification prompt — ``FILTER_SYSTEM`` for messages
                addressed to the bot, ``OVERHEARD_SYSTEM`` for bot-word mentions.

        Returns:
            A label from :func:`parse_label`. Fails open to ``"MEANINGFUL"``
            on any LLM error.
        """
        try:
            response = await self.__llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=text),
            ])
            return parse_label(response.content)
        except Exception as err:  # noqa: BLE001 — fail open: a classifier outage must not mean silence
            logger.warning("Meaningless filter failed, failing open (MEANINGFUL): %s", err)
            return Verdict.MEANINGFUL
