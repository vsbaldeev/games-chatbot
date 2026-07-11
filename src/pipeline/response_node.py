"""ResponseNode — personality LLM that turns worker facts into a chat reply."""

import datetime
import re

from langchain_core.messages import AIMessage, HumanMessage

from src import config, log
from src.agent import needs_russian_correction, normalize_homoglyphs
from src.config.prompts import SHORTS_TRIGGER_INSTRUCTION
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
    parts = ["Что я знаю об участниках чата:"]
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


def build_activity_history_lines(recent_activities: list[tuple[str, float]]) -> list[str]:
    """Return dated activity-history lines, skipping the newest (already-shown) entry.

    Args:
        recent_activities: ``(phrase, posted_at)`` pairs, newest first, as
            stored in ``AssembledContext.bot_recent_activities``. The first
            entry is skipped — it is the same one already rendered as
            ``[Прямо сейчас ты]``/``[Недавно ты]`` by the caller.

    Returns:
        Prompt lines for the remaining dated history, with a trailing blank
        line, or an empty list when fewer than two entries exist.
    """
    history = recent_activities[1:]
    if not history:
        return []
    now = datetime.datetime.now(calendar_ru.MOSCOW_TZ)
    parts = ["[Чем ты занимался в последние дни]:"]
    parts.extend(
        f"- {calendar_ru.describe_relative_day(posted_at, now)} — {phrase}"
        for phrase, posted_at in history
    )
    parts.append("")
    return parts


def build_bot_life_lines(context) -> list[str]:
    """Return formatted bot-canon and current-activity lines for the response prompt.

    Args:
        context: AssembledContext dict or None.

    Returns:
        Prompt lines for relevant canon facts, relevant past episodes, the
        current-activity line and dated activity history, each block only
        present when data exists; empty list when there is no bot canon to
        show (e.g. empty store).
    """
    context = context or {}
    parts: list[str] = []
    facts = context.get("bot_self_facts") or []
    if facts:
        parts.append("[Твоя жизнь — фоновый канон]:")
        parts.extend(f"- {fact}" for fact in facts)
        parts.append("")
    episodes = context.get("bot_self_episodes") or []
    if episodes:
        parts.append("[Твои прошлые истории по теме]:")
        parts.extend(episodes)
        parts.append("")
    activity = context.get("bot_current_activity")
    if activity:
        phrase, freshness = activity
        label = "Прямо сейчас ты" if freshness == "fresh" else "Недавно ты"
        parts.append(f"[{label}]: {phrase}")
        parts.append("")
    parts += build_activity_history_lines(context.get("bot_recent_activities") or [])
    return parts


def build_trigger_line(
    username: str, user_input: str, media_type: str, replied_to: dict | None,
    response_trigger: str = "explicit",
) -> str:
    """Build the final user-turn line, marking media so the model reacts to it.

    Plain text renders as ``@username: text``. Photo/voice/video frame
    ``user_input`` as a description to *react* to, not retell — the chat
    already sees the original. A YouTube Shorts trigger inverts that: nobody
    has watched the video, so the model must retell it and summarize the
    audience reaction from the top comments.

    Args:
        username: Sender's username (without ``@``).
        user_input: The user's words for ``text``, or a vision/transcript
            description for media.
        media_type: ``"text"``, ``"photo"``, ``"voice"``, ``"video_note"``
            or ``"video"``.
        replied_to: The message being replied to, or ``None``.
        response_trigger: Routing trigger; ``"youtube_short"`` selects the
            retell-and-comments-summary framing.

    Returns:
        The trigger line to append as the final human turn.
    """
    speaker = f"@{username}"
    if replied_to:
        speaker = f"{speaker} (↳ {row_speaker(replied_to)})"
    if response_trigger == "youtube_short":
        return f"{speaker} {SHORTS_TRIGGER_INSTRUCTION}:\n{user_input}"
    label = MEDIA_TRIGGER_LABELS.get(media_type)
    if label:
        return (
            f"{speaker} прислал {label}. Ниже — его описание для тебя "
            f"(не дословные слова автора; оригинал в чате все и так видят). "
            f"Отреагируй и пошути, не пересказывай. "
            f"Описание может ошибаться в именах и названиях: не строй шутку "
            f"целиком на конкретном имени или названии, если его не "
            f"подтверждает подпись или разговор:\n{user_input}"
        )
    return f"{speaker}: {user_input}"


def build_recent_history_lines(
    context, response_trigger: str, has_thread_history: bool
) -> tuple[list[str], dict | None]:
    """Return recent-history and replied-to prompt lines, plus the replied-to row.

    Args:
        context: AssembledContext dict or None.
        response_trigger: Routing trigger; ``"random"``/``"youtube_short"``
            trim the recent-history slice further (see :func:`build_response_input`).
        has_thread_history: ``True`` when per-thread turn history is available;
            suppresses recent chat history to avoid double-context.

    Returns:
        Tuple of ``(prompt lines, replied_to)`` — the replied-to row is
        returned alongside so the caller can pass it to :func:`build_trigger_line`
        without recomputing it.
    """
    recent = ((context or {}).get("recent_history") or [])[:RECENT_FILL_LIMIT]
    if response_trigger in ("random", "youtube_short"):
        recent = recent[:RANDOM_TRIGGER_CONTEXT_LIMIT]
    parts: list[str] = []
    if recent and not has_thread_history:
        parts.append("Недавние сообщения чата:")
        parts.extend(render_row(row) for row in reversed(recent))
        parts.append("")

    replied_to = (context or {}).get("replied_to")
    if replied_to:
        recent_ids = {row["message_id"] for row in recent}
        if replied_to["message_id"] not in recent_ids:
            parts.append("Сообщение, на которое отвечают:")
            parts.append(render_row(replied_to))
            parts.append("")
    return parts, replied_to


def build_response_input(
    username: str,
    user_input: str,
    worker_output: str,
    context,
    response_trigger: str = "explicit",
    has_thread_history: bool = False,
    media_type: str = "text",
    is_bot_insult: bool = False,
    worker_tools_used: bool = False,
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
        worker_tools_used: ``True`` when the worker actually ran a tool;
            selects the tool-verified data frame instead of the unverified
            context-derived frame.

    Returns:
        Prompt string ready to pass as the final human turn to the response LLM.
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [f"Текущая дата и время: {now}", ""]
    parts += build_user_facts_lines(context)
    parts += build_asking_user_tag_lines(context, username)
    parts += build_mentioned_tags_lines(context)
    parts += build_bot_life_lines(context)

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
        if worker_tools_used:
            parts.append(f"[Собранные данные (проверено через инструменты)]:\n{worker_output}\n")
        else:
            parts.append(
                f"[Данные из контекста разговора (во внешних источниках НЕ проверялись)]:\n{worker_output}\n"
            )

    if is_bot_insult:
        parts.append(
            "[Это сообщение — наезд на тебя. Не отмалчивайся и не обижайся: "
            "ответь дерзкой, хлёсткой подколкой. Правила: бей по самому наезду, "
            "а не по больным местам человека; держи примерно тот же уровень грубости, "
            "что и он — не жёстче; один удар — и всё: без встречных вопросов "
            "и без приглашений продолжить перепалку.]\n"
        )

    parts.append(build_trigger_line(username, user_input, media_type, replied_to, response_trigger))
    return "\n".join(parts)


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
            worker_tools_used=bool(state.get("worker_tools_used")),
        )
        messages = past_messages + [HumanMessage(content=enriched)]
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
