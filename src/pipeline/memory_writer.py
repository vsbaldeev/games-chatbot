"""
MemoryWriter — fires fact extraction and upsert in the background.

Runs in two modes:
  - Active (bot replied): fed the full exchange (user message + bot reply)
    plus the recent conversation context from state["context"].
  - Passive (no reply): fed the user message alone — the bot "overheard" it.

The module-level extract_and_save is also called directly from the router
for plain text messages that don't trigger a bot response.
Cross-user facts are extracted automatically when @mentions are present.

Source rules:
  - Only the user's own words are evidence. The bot's reply is context for
    understanding the message, never a fact source.
  - Input framing follows source_kind: voice transcripts are labelled as the
    user's spoken words; photo/video descriptions are explicitly marked as
    NOT the user's words (the image content is not a fact about the poster).
  - Cross-user claims pass a sincerity rule (banter and insults are not
    facts) and are stored with a «по словам @X, …» attribution prefix.

Deduplication uses cosine similarity between fastembed vectors rather than
LLM judgement. A duplicate refreshes the existing fact's updated_at instead
of inserting a new row. Facts untouched for 90 days are deleted by the
nightly cleanup job (user_memories.cleanup_stale), counters included.

MEMORY_MODEL is a reasoning model, so every extraction call disables
reasoning (reasoning_effort="none") — otherwise the whole token budget is
burned inside a <think> block and no answer is ever produced. As a backstop,
_parse_facts strips any think blocks first.

Output is one fact per line rather than a JSON array: small models reliably
emit bare, unquoted list items (e.g. "[fact one, fact two]"), which breaks
JSON parsing and silently discards every fact in the batch. Line-splitting
has no quoting failure mode.
"""

import asyncio
import re
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import achievements, config, log
from src.agent import ainvoke_with_backoff
from src.agent.middleware import strip_thinking
from src.config.prompts import (
    CROSS_USER_EXTRACTION_SYSTEM,
    EXTRACTION_SYSTEM,
    MEDIA_DESCRIPTION_RULE,
)
from src.pipeline.state import BotState
from src.store import embedder, user_memories

logger = log.get_logger(__name__)

# Serialises concurrent _dedup_and_save calls for the same user so that the
# check-then-insert window cannot be observed by a second concurrent task.
user_dedup_locks: dict[tuple[int, int], asyncio.Lock] = defaultdict(asyncio.Lock)

# Bounds how many extraction calls hit the shared MEMORY_MODEL token bucket at
# once. Fact extraction fires fire-and-forget per message plus once per @mention,
# so without a cap a burst stampedes the model's tokens-per-minute limit. Excess
# tasks queue on the semaphore instead of all racing for the same 429.
MEMORY_CALL_CONCURRENCY = 2
memory_call_semaphore = asyncio.Semaphore(MEMORY_CALL_CONCURRENCY)
MAX_NEW_FACTS = 3
MIN_PASSIVE_LENGTH = 20
SIMILARITY_THRESHOLD = 0.85

MENTION_RE = re.compile(r"@(\w+)", re.UNICODE)

CONTEXT_SNIPPET_LIMIT = 10
CONTEXT_MSG_CHAR_LIMIT = 120

def format_user_message_line(username: str, user_message: str, source_kind: str) -> str:
    """Frame the user's message so the extractor knows what the text really is.

    Args:
        username: Sender's username (without ``@``).
        user_message: Message text, transcript or media description.
        source_kind: ``"text"``, ``"voice"`` or ``"media_description"``.

    Returns:
        The labelled user-message line: transcripts are the user's words and
        say so; media descriptions are explicitly marked as not their words.
    """
    if source_kind == "voice":
        return f"Расшифровка голосового сообщения @{username}: {user_message}"
    if source_kind == "media_description":
        return (
            f"Описание картинки/видео, которое прислал @{username} "
            f"(это НЕ его слова): {user_message}"
        )
    return f"@{username}: {user_message}"


def _format_recent_context(recent: list[dict]) -> str:
    """Format the last N messages as a readable conversation snippet."""
    lines = []
    for row in recent[-CONTEXT_SNIPPET_LIMIT:]:
        username = row.get("username", "?")
        content = (row.get("content") or "")[:CONTEXT_MSG_CHAR_LIMIT]
        lines.append(f"@{username}: {content}")
    return "\n".join(lines)


def make_extraction_llm() -> ChatGroq:
    """Return the fact-extraction LLM with reasoning disabled.

    MEMORY_MODEL is a reasoning model; without ``reasoning_effort="none"`` it
    spends the whole max_tokens budget inside a ``<think>`` block and the
    answer never appears, so extraction silently yields zero facts.

    Returns:
        Configured ``ChatGroq`` instance for fact extraction.
    """
    return ChatGroq(
        model=config.MEMORY_MODEL, api_key=config.GROQ_API_KEY,
        temperature=0.2, max_tokens=256, max_retries=0,
        reasoning_effort="none",
    )


FACT_LINE_STRIP_RE = re.compile(r"^[\s\-*•\d.)]+")
NO_FACTS_SENTINEL = "NONE"


def _parse_facts(raw: str) -> list[str]:
    """Parse one-fact-per-line extraction output into a list of facts.

    Args:
        raw: Raw LLM output, one fact per line, or the ``NONE`` sentinel
            when nothing distinctive was learned.

    Returns:
        Non-empty fact strings with leading bullets/numbering stripped.
    """
    cleaned = strip_thinking(raw)
    facts = []
    for line in cleaned.splitlines():
        fact = FACT_LINE_STRIP_RE.sub("", line).strip()
        if fact and fact.upper() != NO_FACTS_SENTINEL:
            facts.append(fact)
    return facts


async def _extract_facts(
    *, username: str, user_message: str, bot_reply: str, existing: list[str],
    recent_history: list[dict] | None = None, source_kind: str = "text",
) -> list[str]:
    """Call the LLM to extract new facts about the user from the given exchange.

    Args:
        username: Sender's username (without ``@``).
        user_message: Message text, voice transcript or media description.
        bot_reply: The bot's reply, or empty when the bot was not addressed.
        existing: Facts already stored for the user.
        recent_history: Recent chat rows for referent disambiguation.
        source_kind: What ``user_message`` really is — ``"text"``,
            ``"voice"`` or ``"media_description"``; frames the prompt so a
            meme description is never attributed as the poster's own words.

    Returns:
        Newly extracted fact strings (possibly empty).
    """
    existing_block = "\n".join(f"- {fact}" for fact in existing) if existing else "(none)"
    user_line = format_user_message_line(username, user_message, source_kind)
    exchange = (
        f"Exchange:\n{user_line}\nBot: {bot_reply}"
        if bot_reply
        else f"Message (bot was not addressed):\n{user_line}"
    )
    context_section = (
        f"Recent conversation context:\n{_format_recent_context(recent_history)}\n\n"
        if recent_history else ""
    )
    media_rule = f"{MEDIA_DESCRIPTION_RULE}\n\n" if source_kind == "media_description" else ""
    prompt = (
        f"User: @{username}\nExisting facts:\n{existing_block}\n\n"
        f"{media_rule}{context_section}{exchange}\n\nNew facts to add (one per line):"
    )
    llm = make_extraction_llm()
    async with memory_call_semaphore:
        result = await ainvoke_with_backoff(
            llm, [SystemMessage(content=EXTRACTION_SYSTEM), HumanMessage(content=prompt)],
        )
    return _parse_facts(result.content.strip())


async def _dedup_and_save(
    *, chat_id: int, user_id: int, username: str, new_facts: list[str]
) -> None:
    if not new_facts:
        return
    async with user_dedup_locks[(chat_id, user_id)]:
        await _check_and_insert(chat_id=chat_id, user_id=user_id, username=username, new_facts=new_facts)


async def _check_and_insert(
    *, chat_id: int, user_id: int, username: str, new_facts: list[str]
) -> None:
    to_insert_facts: list[str] = []
    to_insert_embeddings: list[list[float]] = []
    for fact in new_facts[:MAX_NEW_FACTS]:
        fact_embedding = await embedder.embed(fact)
        matched_id = await user_memories.find_similar_fact(
            chat_id=chat_id, user_id=user_id,
            embedding=fact_embedding, threshold=SIMILARITY_THRESHOLD,
        )
        if matched_id is not None:
            await user_memories.refresh_updated_at(matched_id)
        else:
            to_insert_facts.append(fact)
            to_insert_embeddings.append(fact_embedding)
    if to_insert_facts:
        await user_memories.upsert_facts(
            chat_id=chat_id, user_id=user_id, username=username,
            facts=to_insert_facts, embeddings=to_insert_embeddings,
        )
        logger.debug("Saved %d facts for @%s in chat %s", len(to_insert_facts), username, chat_id)


async def extract_and_save(
    *, chat_id: int, user_id: int, username: str, user_message: str, bot_reply: str = "",
    recent_history: list[dict] | None = None, source_kind: str = "text",
) -> None:
    """Extract facts about the sender and any @mentioned users. Safe to use with create_task.

    Args:
        chat_id: Chat the message belongs to.
        user_id: Sender's user id.
        username: Sender's username.
        user_message: Message text, voice transcript or media description.
        bot_reply: The bot's reply, or empty for passive extraction.
        recent_history: Recent chat rows for referent disambiguation.
        source_kind: What ``user_message`` really is — ``"text"``, ``"voice"``
            or ``"media_description"``.
    """
    try:
        existing = await user_memories.get_facts(chat_id=chat_id, user_id=user_id)
        new_facts = await _extract_facts(
            username=username, user_message=user_message,
            bot_reply=bot_reply, existing=existing,
            recent_history=recent_history, source_kind=source_kind,
        )
        await _dedup_and_save(
            chat_id=chat_id, user_id=user_id, username=username, new_facts=new_facts,
        )
    except Exception as err:
        logger.warning("Memory extraction failed for @%s: %s", username, err)
    await _extract_for_mentions(chat_id=chat_id, sender_username=username, user_message=user_message)


CROSS_USER_SINCERITY_RULE = (
    "Это чат друзей, где принято подкалывать друг друга. Оскорбления, подколки, "
    "преувеличения и шутки про упомянутого пользователя — НЕ факты; из них ничего "
    "не извлекай. Извлекай только нейтральные проверяемые утверждения "
    "(купил игру, был в отпуске, выиграл матч)."
)


async def _extract_facts_about(
    *, chat_id: int, user_id: int, username: str,
    observation: str, observer_username: str,
) -> None:
    """Extract facts about an @mentioned user from someone else's message.

    Second-hand claims pass a sincerity rule (banter is not a fact) and the
    survivors are stored with a «по словам @X, …» attribution prefix so every
    downstream prompt sees their epistemic status.

    Args:
        chat_id: Chat the observation was posted in.
        user_id: The mentioned user the facts are about.
        username: The mentioned user's username.
        observation: The full message mentioning the user.
        observer_username: Who made the claim.
    """
    try:
        existing = await user_memories.get_facts(chat_id=chat_id, user_id=user_id)
        existing_block = "\n".join(f"- {fact}" for fact in existing) if existing else "(none)"
        prompt = (
            f"Пользователь: @{username}\nИзвестные факты:\n{existing_block}\n\n"
            f"{CROSS_USER_SINCERITY_RULE}\n\n"
            f"Наблюдение от @{observer_username}: {observation}\n\n"
            f"Что это говорит нам о @{username}? Новые факты (по одному на строку):"
        )
        llm = make_extraction_llm()
        async with memory_call_semaphore:
            result = await ainvoke_with_backoff(
                llm, [SystemMessage(content=CROSS_USER_EXTRACTION_SYSTEM), HumanMessage(content=prompt)],
            )
        new_facts = [
            f"по словам @{observer_username}, {fact}"
            for fact in _parse_facts(result.content.strip())
        ]
        await _dedup_and_save(
            chat_id=chat_id, user_id=user_id, username=username, new_facts=new_facts,
        )
    except Exception as err:
        logger.warning("Cross-user extraction failed for @%s: %s", username, err)


async def _extract_for_mentions(
    *, chat_id: int, sender_username: str, user_message: str
) -> None:
    mentioned = {mention.lower() for mention in MENTION_RE.findall(user_message)}
    mentioned.discard(sender_username.lower())
    if not mentioned:
        return

    stripped = re.sub(r"@\w+", "", user_message).strip()
    if len(stripped) < MIN_PASSIVE_LENGTH:
        return

    try:
        members = await achievements.get_chat_members(chat_id)
    except Exception as err:
        logger.warning("Could not fetch members for cross-user extraction: %s", err)
        return

    username_map = {uname.lower(): (uid, uname) for uid, uname in members}
    for mentioned_lower in mentioned:
        if mentioned_lower not in username_map:
            continue
        uid, original_username = username_map[mentioned_lower]
        asyncio.create_task(_extract_facts_about(
            chat_id=chat_id,
            user_id=uid,
            username=original_username,
            observation=user_message,
            observer_username=sender_username,
        ))


def source_kind_for_media_type(media_type: str) -> str:
    """Map a message's media type onto the extractor's source kind.

    Args:
        media_type: The incoming message's media type.

    Returns:
        ``"text"`` for typed messages, ``"voice"`` for pure speech, and the
        conservative ``"media_description"`` for everything else (photos and
        videos mix model-generated descriptions with transcripts).
    """
    if media_type == "text":
        return "text"
    if media_type == "voice":
        return "voice"
    return "media_description"


class MemoryWriter:
    """Fires background fact-extraction and upsert; returns immediately."""

    async def __call__(self, state: BotState) -> dict:
        """Fire background fact extraction; returns immediately."""
        msg = state["incoming"]
        response = state.get("response") or ""
        user_message = msg["processed_text"] or msg["raw_text"] or ""

        if msg.get("is_forwarded"):
            return {}

        passive = not response.strip()
        if passive and len(user_message.strip()) < MIN_PASSIVE_LENGTH:
            return {}

        assembled = state.get("context")
        recent_history = assembled["recent_history"] if assembled else None

        asyncio.create_task(
            extract_and_save(
                chat_id=msg["chat_id"],
                user_id=msg["user_id"],
                username=msg["username"],
                user_message=user_message,
                bot_reply=response,
                recent_history=recent_history,
                source_kind=source_kind_for_media_type(msg["media_type"]),
            )
        )
        return {}
