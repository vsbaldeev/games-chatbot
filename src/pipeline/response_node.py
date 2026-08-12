"""ResponseNode — personality LLM that turns worker facts into a chat reply."""

import datetime
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage

from src import config, log
from src.agent import needs_russian_correction, normalize_homoglyphs
from src.config.prompts import (
    LINK_REPLY_GROUNDING_INSTRUCTION,
    SHORTS_TRIGGER_INSTRUCTION,
    SOCIAL_LINK_RETELL_INSTRUCTION,
    USER_FACTS_HEADER,
    WEEKLY_ROLES_RULE,
    WORKER_DATA_UNVERIFIED_HEADER,
    WORKER_DATA_VERIFIED_HEADER,
)
from src.life import calendar_ru
from src.pipeline.state import BotState
from src.store import thread_history, unified_messages

logger = log.get_logger(__name__)

RECENT_FILL_LIMIT = 10

# Random (unprompted) triggers get a thin recent-history slice: enough to
# catch an obvious topic mismatch, not enough to drown a spontaneous
# reaction in chat noise.
RANDOM_TRIGGER_CONTEXT_LIMIT = 3

TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")

# Russian labels for the kind of media the triggering message carried. Used to
# mark the current turn as media (not the user's typed words) so the response
# model reacts to it instead of retelling the vision/transcript description.
# build_trigger_line bypasses this dict for "voice" — build_voice_trigger_line
# frames it as speech instead — but persist_thread_turn still consults the
# "voice" entry below to label the persisted thread-history turn, so it stays.
MEDIA_TRIGGER_LABELS = {
    "photo": "фото",
    "voice": "голосовое",
    "video_note": "видеокружок",
    "video": "видео",
}

# Label used for the bot's own past messages in rendered history, so the model
# recognises them as its own turns instead of treating them as another
# participant and @mentioning itself.
SELF_SPEAKER = "Ты (бот)"


def row_speaker(row: dict) -> str:
    """Return the speaker label for a message row.

    The bot's own messages (identified by ``user_id``) are labelled ``Ты (бот)``
    so the model never mistakes them for another participant; everyone else is
    shown as ``@username``.

    Args:
        row: Message row dict with ``user_id`` and ``username`` keys.

    Returns:
        ``"Ты (бот)"`` for the bot's own messages, otherwise ``"@username"``.
    """
    if row.get("user_id") == config.BOT_ID:
        return SELF_SPEAKER
    return f"@{row['username']}"


def strip_markdown(text: str) -> str:
    """Strip common Markdown formatting characters from text.

    Removes bold, italic, and table-separator lines so the output reads as
    plain chat text rather than formatted markup.

    Args:
        text: Input string that may contain Markdown.

    Returns:
        Plain text with bold/italic markers removed and table separators dropped.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text, flags=re.DOTALL)
    lines = [line for line in text.splitlines() if not TABLE_SEP_RE.match(line)]
    return "\n".join(lines)


def render_row(row: dict) -> str:
    """Format a message row as ``speaker [переслал] [media_type]: content``.

    The speaker is ``@username`` for other participants and ``Ты (бот)`` for the
    bot's own past messages (see :func:`row_speaker`). Forwarded rows carry a
    ``[переслал]`` marker so LLM prompts can tell shared channel content from
    the participant's own words.

    Args:
        row: Message dict with ``user_id``, ``username``, ``media_type``, and
            ``content`` keys; ``is_forwarded`` is optional (absent means own words).

    Returns:
        Formatted string representation of the message.
    """
    media_type = row["media_type"]
    content = unified_messages.display_media_content(media_type, row["content"])
    forwarded_label = " [переслал]" if row.get("is_forwarded") else ""
    media_label = f" [{media_type}]" if media_type != "text" else ""
    return f"{row_speaker(row)}{forwarded_label}{media_label}: {content}"


def build_past_messages(history: list[dict]) -> list[HumanMessage | AIMessage]:
    """Convert thread-history records into LangChain message objects.

    Args:
        history: List of dicts with ``role`` (``"human"`` or ``"ai"``) and
            ``content`` keys, ordered oldest-first.

    Returns:
        List of ``HumanMessage`` and ``AIMessage`` instances.
    """
    result: list[HumanMessage | AIMessage] = []
    for entry in history:
        if entry["role"] == "human":
            result.append(HumanMessage(content=entry["content"]))
        else:
            result.append(AIMessage(content=entry["content"]))
    return result


def build_user_facts_lines(context) -> list[str]:
    """Return formatted user-facts section lines for the response prompt.

    Args:
        context: AssembledContext dict or None.

    Returns:
        List of prompt lines, including a trailing blank line, or empty list
        when no facts are available.
    """
    user_facts = (context or {}).get("user_facts") or {}
    if not user_facts:
        return []
    parts = [USER_FACTS_HEADER]
    for uname, facts in user_facts.items():
        parts.append(f"@{uname}: {'; '.join(facts)}")
    parts.append("")
    return parts


def build_asking_user_tag_lines(context, username: str) -> list[str]:
    """Return the asker's own weekly-role lines for the response prompt.

    Args:
        context: AssembledContext dict or None.
        username: Sender's username (without ``@``).

    Returns:
        Prompt lines describing the sender's role and why it was assigned,
        with a trailing blank line, or an empty list when they have no role.
    """
    tag_info = (context or {}).get("asking_user_tag")
    if not tag_info:
        return []
    lines = [f"Роль недели для @{username}: {tag_info['tag']}"]
    reason = tag_info.get("reason")
    if reason:
        lines.append(f"За что выдана: {reason}")
    lines.append("")
    return lines


def build_mentioned_tags_lines(context) -> list[str]:
    """Return weekly-role lines for other members the question @mentions.

    Lets the bot explain why another member got their role, using the stored
    justification rather than improvising.

    Args:
        context: AssembledContext dict or None.

    Returns:
        Prompt lines naming each mentioned member's role and why it was
        assigned, with a trailing blank line, or an empty list when none apply.
    """
    mentioned_tags = (context or {}).get("mentioned_tags") or {}
    if not mentioned_tags:
        return []
    lines = ["Роли недели других участников:"]
    for username, tag_info in mentioned_tags.items():
        reason = tag_info.get("reason")
        suffix = f" — {reason}" if reason else ""
        lines.append(f"@{username}: {tag_info['tag']}{suffix}")
    lines.append("")
    return lines


def build_trigger_line(
    username: str, user_input: str, media_type: str, replied_to: dict | None,
    response_trigger: str = "explicit", voice_low_confidence: bool = False,
) -> str:
    """Build the final user-turn line, marking media so the model reacts to it.

    Plain text renders as ``@username: text``. Voice is framed as the
    person's own spoken words (a transcript, not a description) — joking is
    conditional on the words themselves giving reason to. Photo/video_note/
    video frame ``user_input`` as a description to *react* to, not retell —
    the chat already sees the original; joking there is likewise conditional,
    never mandatory. A YouTube Shorts or social-link trigger always uses
    retell framing: the reply is posted as the caption on the video itself
    (see :func:`src.events.link_repost`), so it is read before anyone
    watches, never after.

    Args:
        username: Sender's username (without ``@``).
        user_input: The user's words for ``text``/``voice``, or a vision/
            fetched-content description for other media/link triggers.
        media_type: ``"text"``, ``"photo"``, ``"voice"``, ``"video_note"``
            or ``"video"``.
        replied_to: The message being replied to, or ``None``.
        response_trigger: Routing trigger; ``"youtube_short"`` and
            ``"social_link"`` select retell framing.
        voice_low_confidence: True when the voice transcript's mean Whisper
            confidence was low (see :func:`src.pipeline.ingester.transcribe_bytes`)
            — adds a note telling the model the transcription may be
            unreliable instead of trusting it as literal content.

    Returns:
        The trigger line to append as the final human turn.
    """
    speaker = f"@{username}"
    if replied_to:
        speaker = f"{speaker} (↳ {row_speaker(replied_to)})"
    if response_trigger == "youtube_short":
        return f"{speaker} {SHORTS_TRIGGER_INSTRUCTION}:\n{user_input}"
    if response_trigger == "social_link":
        return f"{speaker} {SOCIAL_LINK_RETELL_INSTRUCTION}:\n{user_input}"
    if media_type == "voice":
        return build_voice_trigger_line(speaker, user_input, voice_low_confidence)
    label = MEDIA_TRIGGER_LABELS.get(media_type)
    if label:
        return (
            f"{speaker} прислал {label}. Ниже — его описание для тебя "
            f"(не дословные слова автора; оригинал в чате все и так видят — "
            f"очевидное не описывай). Если в описании есть подпись — это "
            f"дословные слова автора, отвечай на неё по существу, а не только "
            f"на картинку. Отреагируй как друг: есть за что зацепиться — "
            f"пошути, преувеличивать можно; нечего — просто ответь по "
            f"существу, не пересказывай. Описание может ошибаться в именах и "
            f"названиях: не строй шутку целиком на конкретном имени или "
            f"названии, если его не подтверждает подпись или разговор:\n{user_input}"
        )
    return f"{speaker}: {user_input}"


def build_voice_trigger_line(speaker: str, user_input: str, low_confidence: bool) -> str:
    """Frame a voice message's trigger line as the person's own spoken words.

    Unlike photo/video framing, a Whisper transcript is not a third-party
    description — it is what the person actually said, so the model is told
    to answer them, not to react to media. Joking is conditional on the
    words themselves, never mandatory.

    Args:
        speaker: Rendered speaker label, already including any reply-chain arrow.
        user_input: The Whisper transcript.
        low_confidence: True when the transcript's mean confidence was low —
            adds a note that the transcription may be unreliable.

    Returns:
        The trigger line to append as the final human turn.
    """
    confidence_note = (
        " Расшифровка может быть неточной из-за шума или плохой слышимости — "
        "если смысл неясен, ответь на то, что разобрал, или переспроси, а не "
        "додумывай за автора."
        if low_confidence else ""
    )
    return (
        f"{speaker} сказал голосовым (ниже — расшифровка его слов, это именно "
        f"то, что он произнёс, а не описание).{confidence_note} Ответь ему по "
        f"существу; шути, только если для этого есть повод в самих "
        f"словах:\n{user_input}"
    )


def build_recent_history_lines(
    context, response_trigger: str, has_thread_history: bool
) -> tuple[list[str], dict | None]:
    """Return recent-history and replied-to prompt lines, plus the replied-to row.

    The replied-to block is skipped only when that message was actually
    rendered in the recent-history block above it — never merely because it
    sits in the recent window. Keying the check on the window instead made
    the two blocks suppress each other on reply chains (recent history
    dropped by ``has_thread_history``, replied-to dropped as "already
    shown"), leaving the model with an unanchored ``(↳ …)`` arrow and no way
    to resolve a short follow-up like «На четвертом».

    When the replied-to row carries ``link_material`` (a direct reply to one
    of the bot's own link-repost messages — see the addressing gate in
    ``src.pipeline.router.MessageRouter``), the persisted ingestion material
    is injected as a labelled, honesty-guarded block. This happens
    independently of whether the "Сообщение, на которое отвечают:" header
    itself was shown — that header renders the caption text, which
    recent-history may already have shown, but the material is never
    rendered by anything else, so it must still appear.

    Args:
        context: AssembledContext dict or None.
        response_trigger: Routing trigger; ``"random"``/``"youtube_short"``/
            ``"social_link"`` trim the recent-history slice further (see
            :func:`build_response_input`).
        has_thread_history: ``True`` when per-thread turn history is available;
            suppresses recent chat history to avoid double-context.

    Returns:
        Tuple of ``(prompt lines, replied_to)`` — the replied-to row is
        returned alongside so the caller can pass it to :func:`build_trigger_line`
        without recomputing it.
    """
    recent = ((context or {}).get("recent_history") or [])[:RECENT_FILL_LIMIT]
    if response_trigger in ("random", "youtube_short", "social_link"):
        recent = recent[:RANDOM_TRIGGER_CONTEXT_LIMIT]
    rendered = [] if has_thread_history else recent

    parts: list[str] = []
    if rendered:
        parts.append("Недавние сообщения чата:")
        parts.extend(render_row(row) for row in reversed(rendered))
        parts.append("")

    replied_to = (context or {}).get("replied_to")
    if replied_to:
        rendered_ids = {row["message_id"] for row in rendered}
        if replied_to["message_id"] not in rendered_ids:
            parts.append("Сообщение, на которое отвечают:")
            parts.append(render_row(replied_to))
            parts.append("")
        link_material = replied_to.get("link_material")
        if link_material:
            parts.append(LINK_REPLY_GROUNDING_INSTRUCTION.format(material=link_material))
            parts.append("")
    return parts, replied_to


def build_directive_lines(
    is_bot_insult: bool, wind_down: bool, photo_directive: str | None,
    meme_directive: str | None = None,
) -> list[str]:
    """Assemble the behavioural directive blocks appended before the trigger line.

    Args:
        is_bot_insult: ``True`` when the filter classified the message as an
            insult aimed at the bot; adds a hint telling the model to clap
            back — unless ``wind_down`` is also set (see below).
        wind_down: ``True`` when the engagement gate wants the conversation
            closed; adds a hint to answer in one short phrase and disengage.
            When combined with ``is_bot_insult``, ``filter_node`` set both
            either because the insult replies to the bot's own message
            (``replies_to_bot``), or because the sender's engagement tier had
            already dropped below full — in both cases a mirrored
            counter-insult would fuel a loop the gate is already trying to
            end, so only this softer line fires and the aggressive comeback
            is suppressed.
        photo_directive: Photo-request framing — ``"ack"`` (generation is
            being launched, promise the photo), ``"busy"`` (a selfie is
            already rendering, no second one), ``"refused"`` (wound-down user
            asked for a photo, refuse it explicitly) or None.
        meme_directive: Meme-request framing — ``"refused"`` (wound-down user
            asked for a meme) or None. There is no ``"ack"`` counterpart: an
            accepted meme request answers with the image and no text, so the
            response node never runs for it.

    Returns:
        Directive prompt lines, possibly empty.
    """
    lines: list[str] = []
    if is_bot_insult and not wind_down:
        lines.append(
            "[Это сообщение — наезд на тебя. Не отмалчивайся и не обижайся: "
            "ответь дерзкой, хлёсткой подколкой. Правила: бей по самому наезду, "
            "а не по больным местам человека; держи примерно тот же уровень грубости, "
            "что и он — не жёстче; один удар — и всё: без встречных вопросов "
            "и без приглашений продолжить перепалку.]\n"
        )
    if wind_down:
        lines.append(
            "[Тебе уже надоел этот разговор. Ответь очень коротко — одной фразой, "
            "в своём характере: дай понять, что сворачиваешь болтовню (дела, "
            "работа, некогда). Без встречных вопросов и без приглашений "
            "продолжить.]\n"
        )
    if photo_directive == "ack":
        lines.append(
            "[Тебя просят прислать твоё фото. Ты уже пошёл фоткать — ответь одной "
            "короткой фразой в своём духе, что сейчас сфоткаешь и скинешь. "
            "Не описывай будущее фото и ничего про него не выдумывай.]\n"
        )
    elif photo_directive == "busy":
        lines.append(
            "[Ты уже фоткаешь по предыдущей просьбе. Скажи коротко, что уже этим "
            "занят и фото скоро будет; второй раз фоткать не пойдёшь.]\n"
        )
    elif photo_directive == "refused":
        lines.append(
            "[Тебя просят прислать твоё фото, но фоткаться тебе лень и некогда. "
            "Откажи прямо, в своём характере, без обещаний прислать позже.]\n"
        )
    if meme_directive == "refused":
        lines.append(
            "[Тебя просят скинуть мем, но тебе сейчас не до этого. Откажи "
            "коротко и в своём характере, без обещаний скинуть позже. "
            "Не пересказывай и не выдумывай никаких мемов.]\n"
        )
    return lines


def build_response_input(
    username: str,
    user_input: str,
    worker_output: str,
    context,
    response_trigger: str = "explicit",
    has_thread_history: bool = False,
    media_type: str = "text",
    is_bot_insult: bool = False,
    wind_down: bool = False,
    worker_tools_used: bool = False,
    photo_directive: str | None = None,
    meme_directive: str | None = None,
    voice_low_confidence: bool = False,
) -> str:
    """Assemble the enriched user-turn string for the response LLM.

    Args:
        username: Sender's username (without ``@``).
        user_input: Processed text of the triggering message.
        worker_output: Facts gathered by the worker agent, or empty string.
        context: AssembledContext dict or None.
        response_trigger: ``"explicit"`` when the bot was @mentioned or replied
            to; ``"random"`` for unprompted triggers.
        has_thread_history: ``True`` when per-thread turn history is available;
            suppresses recent chat history to avoid double-context.
        media_type: Media kind of the triggering message; non-text values mark
            the trigger line as a media description to react to (see
            :func:`build_trigger_line`).
        is_bot_insult: ``True`` when the filter classified the message as an
            insult aimed at the bot; adds a hint telling the model to clap back.
        wind_down: ``True`` when the engagement gate wants the conversation
            closed; adds a hint to answer in one short phrase and disengage.
        worker_tools_used: ``True`` when the worker actually ran a tool;
            selects the tool-verified data frame instead of the unverified
            context-derived frame.
        meme_directive: Meme-request framing passed to
            :func:`build_directive_lines` — ``"refused"`` or None.
        photo_directive: Photo-request framing passed to
            :func:`build_directive_lines`, or None.
        voice_low_confidence: True when the voice transcript's mean Whisper
            confidence was low; passed straight through to
            :func:`build_trigger_line`.

    Returns:
        Prompt string ready to pass as the final human turn to the response LLM.
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [f"Текущая дата и время: {now}", ""]
    parts += build_user_facts_lines(context)
    role_lines = build_asking_user_tag_lines(context, username) + build_mentioned_tags_lines(context)
    if role_lines:
        parts.append(WEEKLY_ROLES_RULE)
        parts += role_lines

    # Skip recent history when thread history is present (thread turns already
    # provide conversational context, group chat would just confuse the model).
    # Random and Shorts triggers keep a thin slice — enough to catch topic
    # mismatch without turning a spontaneous reaction into a reply to the
    # discussion.
    history_lines, replied_to = build_recent_history_lines(
        context, response_trigger, has_thread_history
    )
    parts += history_lines

    if worker_output:
        header = WORKER_DATA_VERIFIED_HEADER if worker_tools_used else WORKER_DATA_UNVERIFIED_HEADER
        parts.append(f"{header}\n{worker_output}\n")

    parts += build_directive_lines(is_bot_insult, wind_down, photo_directive, meme_directive)

    parts.append(
        build_trigger_line(
            username, user_input, media_type, replied_to, response_trigger,
            voice_low_confidence=voice_low_confidence,
        )
    )
    return "\n".join(parts)


def resolve_photo_directive(state: BotState) -> str | None:
    """Derive the photo-request directive from the filter's state flags.

    Args:
        state: Current pipeline state.

    Returns:
        ``"ack"`` for an accepted photo request, ``"busy"`` when a selfie was
        already rendering, ``"refused"`` for a wound-down photo request, or
        None when the message is not photo-related.
    """
    if state.get("photo_request"):
        return "busy" if state.get("photo_in_flight") else "ack"
    if state.get("wind_down") and state.get("filter_verdict") == "PHOTO_REQUEST":
        return "refused"
    return None


def resolve_meme_directive(state: BotState) -> str | None:
    """Derive the meme-request directive from the filter's state flags.

    Args:
        state: Current pipeline state.

    Returns:
        ``"refused"`` when a wound-down user asked for a meme, else None.
        An accepted meme request never reaches here — the node short-circuits
        to an empty response and the events layer sends the image alone.
    """
    if state.get("wind_down") and state.get("filter_verdict") == "MEME_REQUEST":
        return "refused"
    return None


async def persist_thread_turn(state: BotState, response_text: str) -> None:
    """Append the finished exchange to thread history.

    Must be called with the reply the chat actually saw — after language
    correction when it runs. Flat (non-reply) exchanges are stored under the
    prospective chain root (the triggering message id), so a follow-up reply
    to the bot's answer derives a thread pre-seeded with this exchange;
    reply-chain exchanges keep their derived thread id.

    Args:
        state: Current pipeline state.
        response_text: Final reply text as sent to the chat.
    """
    if not response_text.strip():
        return
    msg = state["incoming"]
    if state.get("is_flat_thread"):
        thread_id = thread_history.thread_id_for_root(msg["chat_id"], msg["message_id"])
    else:
        thread_id = state.get("thread_id") or str(msg["chat_id"])
    user_input = msg["processed_text"] or msg["raw_text"] or ""
    media_label = MEDIA_TRIGGER_LABELS.get(msg["media_type"])
    speaker = f"@{msg['username']}"
    human_content = (
        f"{speaker} [{media_label}]: {user_input}" if media_label
        else f"{speaker}: {user_input}"
    )
    await thread_history.append_turn(
        thread_id=thread_id,
        chat_id=msg["chat_id"],
        human_content=human_content,
        ai_content=strip_markdown(response_text),
    )


def log_response_input(past_messages: list, enriched: str) -> None:
    """Dump the exact input the response LLM is about to see, at DEBUG level.

    Absurd replies are usually caused by something in the assembled prompt
    (a stale fact, a poisoned thread turn, a mis-framed trigger line), and
    that assembly is ephemeral — without this dump there is no way to see
    it after the fact. Thread history turns are logged as one-line excerpts
    (their full text lives in the ``thread_history`` table); the enriched
    final turn is logged verbatim because it exists nowhere else.

    Args:
        past_messages: Thread-history turns preceding the final human turn.
        enriched: The assembled final human turn passed to the LLM.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for position, message in enumerate(past_messages, start=1):
        speaker = "human" if isinstance(message, HumanMessage) else "ai"
        logger.debug(
            "Response LLM history %d/%d (%s): %s",
            position, len(past_messages), speaker, log.snippet(message.content),
        )
    logger.debug("Response LLM final turn:\n%s", enriched)


class ResponseNode:
    """Generates the final personality-driven reply from gathered worker facts."""

    def __init__(self, agent) -> None:
        """Initialize ResponseNode.

        Args:
            agent: Agent instance used to invoke the response LLM.
        """
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        """Generate a response and persist the turn to thread history.

        Args:
            state: Current pipeline state with incoming message, context, and
                optional worker output.

        Returns:
            Dict with ``response`` and ``response_messages`` keys; the latter
            carries the assembled LangChain message list for the correction node.
        """
        if state.get("meme_request"):
            # The meme is the whole reply: a «держи мем» line would be filler,
            # and a generated line about an image this model never sees is the
            # stacking failure the 2026-08-07 absurdity work removed. Returning
            # empty also skips the response LLM entirely.
            return {"response": "", "response_messages": []}

        msg = state["incoming"]

        if state.get("is_flat_thread"):
            # Flat mentions are answered from recent chat context; the old
            # flat bucket held only the bot's own stale exchanges.
            past_messages: list[HumanMessage | AIMessage] = []
        else:
            thread_id = state.get("thread_id") or str(msg["chat_id"])
            history = await thread_history.get_history(
                thread_id=thread_id, limit=config.MAX_HISTORY_MESSAGES
            )
            past_messages = build_past_messages(history)

        user_input = msg["processed_text"] or msg["raw_text"] or ""
        media_type = msg["media_type"]
        enriched = build_response_input(
            msg["username"],
            user_input,
            state.get("worker_output") or "",
            state.get("context"),
            state.get("response_trigger") or "explicit",
            has_thread_history=bool(past_messages),
            media_type=media_type,
            is_bot_insult=bool(state.get("is_bot_insult")),
            wind_down=bool(state.get("wind_down")),
            worker_tools_used=bool(state.get("worker_tools_used")),
            photo_directive=resolve_photo_directive(state),
            meme_directive=resolve_meme_directive(state),
            voice_low_confidence=bool(state.get("voice_low_confidence")),
        )
        messages = past_messages + [HumanMessage(content=enriched)]
        log_response_input(past_messages, enriched)
        response_text = normalize_homoglyphs(await self.__generate(messages))

        # Foreign-script responses are persisted by LanguageCorrectionNode
        # after the retry, so history stores the reply the chat actually saw.
        if not needs_russian_correction(response_text or ""):
            await persist_thread_turn(state, response_text)

        return {"response": response_text, "response_messages": messages}

    async def __generate(self, messages: list) -> str:
        """Delegate to the response agent.

        Args:
            messages: Assembled message list (history + human turn, no system prompt —
                the executor prepends it internally).

        Returns:
            Reply text from the agent. Empty string if the agent returned nothing.

        Raises:
            ContextLengthError: If the prompt exceeds the model's context window.
            DailyLimitError: If all models have exhausted their daily token quota.
            RateLimitError: If rate-limit retries are exhausted on all models.
        """
        return await self.__agent.invoke_response(messages)
