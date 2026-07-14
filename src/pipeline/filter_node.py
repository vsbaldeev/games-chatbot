"""
MeaninglessFilterNode — third node in the LangGraph pipeline.

Uses an LLM to classify the message (text or transcribed media) depending on
how it entered the pipeline:
  - Addressed messages (@mention / reply to the bot) are classified as
    MEANINGLESS (emoji reaction, no reply), BOT_INSULT (insult ladder) or
    MEANINGFUL (normal reply). When the message replies to an earlier stored
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
    A MEANINGLESS verdict on a text that looks like a question or an
    imperative request (question mark, leading interrogative, or a request
    verb like «переведи»/«расскажи») is overridden to MEANINGFUL: a question
    or request is never meaningless.
  - Overheard messages routed by the router's bot-word check
    (response_trigger="insult_check") are classified with the last few chat
    messages as context, so the model can tell this bot from game bots,
    other Telegram bots and people playing «как бот». A BOT_INSULT verdict
    acts only after a stronger model confirms it on the same input
    (disagreement or a confirmation error resolves to silence); everything
    else is dropped silently — no emoji reaction, because the bot was never
    addressed and reacting would be noise. The insult counter fact is thus
    recorded only for addressed or double-confirmed insults.
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

Confirmed insults walk the per-user escalation ladder (see ``insult_gate``):
full comeback → canned dismissive one-liner → bored emoji reaction. Every
confirmed insult also increments the «Оскорблял бота N раз» counter fact in
user_memories, which feeds weekly roles and other engagement features.

Emoji reactions and fact writes fire via asyncio.create_task and do not block
the pipeline.
"""

import asyncio
import random
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import ReactionTypeEmoji

from src import config, log
from src.config.prompts import (
    BOT_INSULT_EXAMPLES,
    CRUDE_PRAISE_EXAMPLES,
    FILTER_SYSTEM,
    OVERHEARD_SYSTEM,
)
from src.pipeline import insult_gate
from src.pipeline.ingester import enrich_media_row
from src.pipeline.memory_writer import MIN_PASSIVE_LENGTH, extract_and_save
from src.pipeline.router import is_explicitly_addressed
from src.pipeline.state import BotState
from src.store import unified_messages, user_memories

logger = log.get_logger(__name__)

REACTION_POOL = ["👍", "❤", "🔥", "😁", "👀", "😎", "💯", "🤣", "⚡", "🫡", "🎉"]

# Bored acknowledgement for repeat insulters (IGNORE_TIER) — must stay within
# Telegram's fixed set of allowed reaction emoji.
DISMISSIVE_REACTIONS = ["🥱", "😴", "🗿", "🤨"]

# Canned tier-2 answers: dry, bored, no LLM call — boredom deflates a troll
# better than escalation.
DISMISSIVE_REPLIES = [
    "Опять ты? Скучно.",
    "Это уже было. Придумай что-нибудь новое.",
    "Второй заход, а смешнее не стало.",
    "Настойчиво. Бездарно, но настойчиво.",
    "Я бы обиделся, если бы было на что.",
    "Записал в тетрадку обид. Страница всё ещё пустая.",
    "Материал повторяется, зритель зевает.",
    "Ок. Что-нибудь ещё?",
]

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

# Leading interrogatives that mark a message as a real question even without
# a question mark — Russian and English.
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

# Cap on how much replied-to text is fed to the classifier as context.
REPLIED_TO_CHAR_LIMIT = 500

# Any bare media placeholder ([photo], [sticker], [voice]…) is opaque to the
# classifier — showing the literal token invites a MEANINGLESS verdict on a
# reply that engages with content the classifier cannot see.
PLACEHOLDER_RE = re.compile(r"^\[\w+\]$")

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


def looks_like_request(text: str) -> bool:
    """Cheap deterministic check that a message is a question or imperative request.

    Used to override a MEANINGLESS verdict: a question or request addressed
    to the bot always deserves a reply, however short it is — even when the
    classifier erred because the quoted content was opaque to it.

    Every MEANINGLESS category is a SHORT reaction (laughter, «ок», «бля»,
    emoji, «хз»), so a message with more than ``SUBSTANTIVE_WORD_COUNT``
    non-laughter word tokens is treated as substantive regardless of its
    leading word — this catches long requests like «поищи в интернете, когда…»
    that a weak classifier mislabels and that no leading-word check would save.

    Args:
        text: Raw message text.

    Returns:
        True when the text contains a question mark, has more than
        ``SUBSTANTIVE_WORD_COUNT`` non-laughter words, or its first
        non-laughter word is an interrogative from ``QUESTION_WORDS`` or an
        imperative from ``REQUEST_WORDS``.
    """
    if "?" in text:
        return True
    words = [word for word in re.findall(r"\w+", text.lower()) if not LAUGHTER_RE.fullmatch(word)]
    if len(words) > SUBSTANTIVE_WORD_COUNT:
        return True
    return bool(words) and words[0] in (QUESTION_WORDS | REQUEST_WORDS)


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


class MeaninglessFilterNode:
    """LLM filter separating meaningless reactions, bot insults and real messages."""

    def __init__(self) -> None:
        """Build the classification and confirmation LLMs from configuration."""
        self.__llm = ChatGroq(
            model=config.FILTER_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=10,
            max_retries=0,
        )
        self.__confirm_llm = ChatGroq(
            model=config.INSULT_CONFIRM_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=10,
            max_retries=0,
        )

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

        if state["incoming"]["media_type"] != "text":
            return self.__handle_media(state)

        text = state["incoming"]["raw_text"] or ""
        if not text.strip():
            return {"should_respond": False}

        if state.get("response_trigger") == "insult_check":
            recent = await self.__fetch_recent_context(state["incoming"])
            overheard_input = build_overheard_input(text, recent)
            decision = await self.__classify(overheard_input, OVERHEARD_SYSTEM)
            return await self.__resolve_overheard(state, decision, overheard_input)

        decision = await self.__classify_addressed(state, text)
        return self.__resolve_addressed(state, decision)

    async def __classify_addressed(self, state: BotState, text: str) -> str:
        """Classify an addressed message with reply context and a request override.

        Loads the replied-to message (if any) so the classifier sees what the
        user is reacting to — vision-enriching a photo or sticker row still
        in placeholder form first, so a reply to a meme is classified against
        the actual image content, not an opaque ``[photo]`` token. Then
        refuses to let a question or request be dropped: a MEANINGLESS
        verdict on a text that looks like one is overridden to MEANINGFUL,
        keeping BOT_INSULT verdicts intact.

        Args:
            state: Current pipeline state.
            text: Raw message text.

        Returns:
            One of ``"BOT_INSULT"``, ``"MEANINGLESS"`` or ``"MEANINGFUL"``.
        """
        replied_to = await self.__fetch_replied_to(state["incoming"])
        replied_to = await self.__enrich_replied_media(replied_to, state)
        decision = await self.__classify(build_filter_input(text, replied_to), FILTER_SYSTEM)
        if decision == "MEANINGLESS" and looks_like_request(text):
            logger.info(
                "Filter: message %s is a question/request — overriding MEANINGLESS to MEANINGFUL",
                state["incoming"]["message_id"],
            )
            return "MEANINGFUL"
        return decision

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
            return {}
        message_id = state["incoming"]["message_id"]
        telegram_message = state["incoming"]["update"].message
        if telegram_message is not None and is_explicitly_addressed(
            telegram_message, config.BOT_USERNAME, config.BOT_ID
        ):
            logger.warning(
                "Filter: no Shorts content for message %s, explicit trigger — canned failure reply",
                message_id,
            )
            return {"should_respond": False, "response": random.choice(SHORTS_FAILED_REPLIES)}
        logger.warning("Filter: no Shorts content for message %s, skipping silently", message_id)
        return {"should_respond": False}

    def __handle_media(self, state: BotState) -> dict:
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
            return self.__handle_transcribed_media(state, media_type)
        if state.get("response_trigger") == "explicit":
            logger.warning(
                "Filter: no transcription for %s message %s, explicit trigger — canned failure reply",
                media_type,
                state["incoming"]["message_id"],
            )
            pool = VISION_FAILED_REPLIES if media_type == "photo" else TRANSCRIPTION_FAILED_REPLIES
            return {"should_respond": False, "response": random.choice(pool)}
        logger.warning(
            "Filter: no transcription for %s message %s, skipping",
            media_type,
            state["incoming"]["message_id"],
        )
        asyncio.create_task(self.__send_reaction(state))
        return {"should_respond": False}

    def __handle_transcribed_media(self, state: BotState, media_type: str) -> dict:
        """Pass a successfully transcribed/described media message through.

        A random-trigger photo/video note the vision classifier flagged as
        not a real person (a meme) is dropped instead — silently, as if the
        random roll had simply missed.

        Args:
            state: Current pipeline state.
            media_type: The incoming message's media type.

        Returns:
            State update dict.
        """
        if is_meme_random_trigger(state, media_type):
            logger.info(
                "Filter: random-trigger %s message %s looks like a meme — skipping",
                media_type, state["incoming"]["message_id"],
            )
            return {"should_respond": False}
        return {}

    def __resolve_addressed(self, state: BotState, decision: str) -> dict:
        """Apply the verdict for a message explicitly addressed to the bot.

        Args:
            state: Current pipeline state.
            decision: Classifier verdict.

        Returns:
            State update dict.
        """
        message_id = state["incoming"]["message_id"]
        if decision == "MEANINGLESS":
            logger.info("Filter: Dropping meaningless message %s", message_id)
            asyncio.create_task(self.__send_reaction(state))
            return {"should_respond": False}
        if decision == "BOT_INSULT":
            return self.__resolve_insult(state)
        return {"should_respond": True}

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
            logger.info("Filter: overheard insult confirmation verdict: %s", verdict)
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
            return self.__resolve_insult(state)
        text = msg["raw_text"] or ""
        if len(text.strip()) >= MIN_PASSIVE_LENGTH:
            asyncio.create_task(extract_and_save(
                chat_id=msg["chat_id"],
                user_id=msg["user_id"],
                username=msg["username"],
                user_message=text,
            ))
        return {"should_respond": False}

    def __resolve_insult(self, state: BotState) -> dict:
        """Walk the escalation ladder for a confirmed insult aimed at the bot.

        Records the insult as a counter fact in user_memories (background) and
        picks the response tier: full comeback for the first insult in the
        window, a canned dismissive one-liner for the next couple, and a bored
        emoji reaction beyond that — so one insult entertains the chat but a
        barrage gets starved of attention.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict.
        """
        msg = state["incoming"]
        tier = insult_gate.register_insult(msg["chat_id"], msg["user_id"])
        asyncio.create_task(self.__record_insult(msg))
        if tier == insult_gate.COMEBACK_TIER:
            logger.info("Filter: insult at the bot in message %s — clapping back", msg["message_id"])
            return {"should_respond": True, "is_bot_insult": True}
        if tier == insult_gate.DISMISSIVE_TIER:
            logger.info("Filter: repeat insult in message %s — dismissive reply", msg["message_id"])
            return {"should_respond": False, "response": random.choice(DISMISSIVE_REPLIES)}
        logger.info("Filter: insult barrage in message %s — emoji only", msg["message_id"])
        asyncio.create_task(self.__send_reaction(state, DISMISSIVE_REACTIONS))
        return {"should_respond": False}

    async def __record_insult(self, msg: dict) -> None:
        """Increment the bot-insult counter fact for the insulter.

        Args:
            msg: IncomingMessage dict of the insulting message.
        """
        try:
            await user_memories.upsert_insult_attempt(
                chat_id=msg["chat_id"],
                user_id=msg["user_id"],
                username=msg["username"],
            )
        except Exception as err:
            logger.warning("Failed to record insult fact for @%s: %s", msg["username"], err)

    async def __classify(self, text: str, system_prompt: str) -> str:
        """Classify the message text with the filter LLM.

        Args:
            text: Raw message text, or the reply-context input assembled by
                ``build_filter_input`` for addressed replies.
            system_prompt: Classification prompt — ``FILTER_SYSTEM`` for messages
                addressed to the bot, ``OVERHEARD_SYSTEM`` for bot-word mentions.

        Returns:
            One of ``"BOT_INSULT"``, ``"MEANINGLESS"`` or ``"MEANINGFUL"``.
            Fails open to ``"MEANINGFUL"`` on any LLM error.
        """
        try:
            response = await self.__llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=text),
            ])
            result = response.content.strip().upper()
            if "INSULT" in result:
                return "BOT_INSULT"
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
