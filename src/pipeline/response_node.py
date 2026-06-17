"""ResponseNode — personality LLM that turns worker facts into a chat reply."""

import datetime
import re

from langchain_core.messages import AIMessage, HumanMessage

from src import config, log
from src.pipeline.state import BotState
from src.store import thread_history, unified_messages

logger = log.get_logger(__name__)

RECENT_FILL_LIMIT = 10
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
    """Format a message row as ``speaker [media_type]: content``.

    The speaker is ``@username`` for other participants and ``Ты (бот)`` for the
    bot's own past messages (see :func:`row_speaker`).

    Args:
        row: Message dict with ``user_id``, ``username``, ``media_type``, and
            ``content`` keys.

    Returns:
        Formatted string representation of the message.
    """
    media_type = row["media_type"]
    content = row["content"]
    if media_type == "photo":
        content = unified_messages.display_photo_content(content)
    media_label = f" [{media_type}]" if media_type != "text" else ""
    return f"{row_speaker(row)}{media_label}: {content}"


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


def build_trigger_line(
    username: str, user_input: str, media_type: str, replied_to: dict | None
) -> str:
    """Build the final user-turn line, marking media so the model reacts to it.

    For a plain text message this is just ``@username: text`` (optionally noting
    the message it replies to). For a photo/voice/video the line frames
    ``user_input`` as a description the model must *react* to rather than retell,
    because in the chat everyone already sees the original media.

    Args:
        username: Sender's username (without ``@``).
        user_input: Triggering message text — the user's words for ``text``, or a
            vision/transcript description for media.
        media_type: ``"text"``, ``"photo"``, ``"voice"``, ``"video_note"`` or
            ``"video"``.
        replied_to: The message being replied to, or ``None``.

    Returns:
        The trigger line to append as the final human turn.
    """
    speaker = f"@{username}"
    if replied_to:
        speaker = f"{speaker} (↳ {row_speaker(replied_to)})"
    label = MEDIA_TRIGGER_LABELS.get(media_type)
    if label:
        return (
            f"{speaker} прислал {label}. Ниже — его описание для тебя "
            f"(не дословные слова автора; оригинал в чате все и так видят). "
            f"Отреагируй и пошути, не пересказывай:\n{user_input}"
        )
    return f"{speaker}: {user_input}"


def build_response_input(
    username: str,
    user_input: str,
    worker_output: str,
    context,
    response_trigger: str = "explicit",
    has_thread_history: bool = False,
    media_type: str = "text",
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

    Returns:
        Prompt string ready to pass as the final human turn to the response LLM.
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [f"Текущая дата и время: {now}", ""]
    parts += build_user_facts_lines(context)
    parts += build_asking_user_tag_lines(context, username)
    parts += build_mentioned_tags_lines(context)

    recent = ((context or {}).get("recent_history") or [])[:RECENT_FILL_LIMIT]
    # Skip recent history when thread history is present (thread turns already
    # provide conversational context, group chat would just confuse the model)
    # or when the trigger is random (bot should focus only on the triggering media).
    if recent and response_trigger != "random" and not has_thread_history:
        parts.append("Недавние сообщения чата:")
        for row in reversed(recent):
            parts.append(render_row(row))
        parts.append("")

    replied_to = (context or {}).get("replied_to")
    if replied_to:
        recent_ids = {row["message_id"] for row in recent}
        if replied_to["message_id"] not in recent_ids:
            parts.append("Сообщение, на которое отвечают:")
            parts.append(render_row(replied_to))
            parts.append("")

    if worker_output:
        parts.append(f"[Собранные данные]:\n{worker_output}\n")

    parts.append(build_trigger_line(username, user_input, media_type, replied_to))
    return "\n".join(parts)


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
        )
        messages = past_messages + [HumanMessage(content=enriched)]
        response_text = await self.__generate(messages)

        if response_text.strip():
            media_label = MEDIA_TRIGGER_LABELS.get(media_type)
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
