"""
MemoryWriter — fires fact extraction and upsert in the background.

Runs in two modes:
  - Active (bot replied): fed the full exchange (user message + bot reply).
  - Passive (no reply): fed the user message alone — the bot "overheard" it.

The module-level extract_and_save is also called directly from the router
for plain text messages that don't trigger a bot response.
Cross-user facts are extracted automatically when @mentions are present.

Deduplication uses cosine similarity between fastembed vectors rather than
LLM judgement. A duplicate refreshes the existing fact's updated_at instead
of inserting a new row.
"""

import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import achievements, config, log
from src.pipeline.state import BotState
from src.store import embedder, user_memories

logger = log.get_logger(__name__)

MEMORY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_NEW_FACTS = 3
MIN_PASSIVE_LENGTH = 20
SIMILARITY_THRESHOLD = 0.85

MENTION_RE = re.compile(r"@(\w+)", re.UNICODE)

EXTRACTION_SYSTEM = (
    "Ты извлекаешь краткие факты о человеке из одного обмена сообщениями в чате. "
    "Возвращай JSON-массив коротких строк на русском языке (не более 15 слов каждая). "
    "Включай только факты, которых ещё нет или которые обновляют уже известные. "
    "Извлекай только отличительные факты — то, что выделяет этого человека среди других: "
    "игровые предпочтения, привычки, мнения, события, достижения, странности. "
    "Пропускай очевидное и универсальное: язык общения, использование эмодзи, наличие телефона, "
    "написание сообщений — это справедливо для всех участников чата и бесполезно. "
    "Если ничего отличительного не узнано — верни []. Без пояснений, без markdown — только сырой JSON."
)


def _parse_facts(raw: str) -> list[str]:
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        facts = []
        for item in data:
            if isinstance(item, str):
                fact = item.strip()
            elif isinstance(item, dict):
                fact = next((str(value).strip() for value in item.values() if value), "")
            else:
                fact = str(item).strip()
            if fact:
                facts.append(fact)
        return facts
    except json.JSONDecodeError:
        return []


async def _extract_facts(
    *, username: str, user_message: str, bot_reply: str, existing: list[str]
) -> list[str]:
    existing_block = "\n".join(f"- {fact}" for fact in existing) if existing else "(none)"
    exchange = (
        f"Exchange:\n@{username}: {user_message}\nBot: {bot_reply}"
        if bot_reply
        else f"Message (bot was not addressed):\n@{username}: {user_message}"
    )
    prompt = (
        f"User: @{username}\nExisting facts:\n{existing_block}\n\n"
        f"{exchange}\n\nNew facts to add (JSON array):"
    )
    llm = ChatGroq(model=MEMORY_MODEL, api_key=config.GROQ_API_KEY, temperature=0.2, max_tokens=256)
    result = await llm.ainvoke([SystemMessage(content=EXTRACTION_SYSTEM), HumanMessage(content=prompt)])
    return _parse_facts(result.content.strip())


async def _dedup_and_save(
    *, chat_id: int, user_id: int, username: str, new_facts: list[str]
) -> None:
    if not new_facts:
        return
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
    *, chat_id: int, user_id: int, username: str, user_message: str, bot_reply: str = ""
) -> None:
    """Extract facts about the sender and any @mentioned users. Safe to use with create_task."""
    try:
        existing = await user_memories.get_facts(chat_id=chat_id, user_id=user_id)
        new_facts = await _extract_facts(
            username=username, user_message=user_message,
            bot_reply=bot_reply, existing=existing,
        )
        await _dedup_and_save(
            chat_id=chat_id, user_id=user_id, username=username, new_facts=new_facts,
        )
    except Exception as err:
        logger.warning("Memory extraction failed for @%s: %s", username, err)
    await _extract_for_mentions(chat_id=chat_id, sender_username=username, user_message=user_message)


async def _extract_facts_about(
    *, chat_id: int, user_id: int, username: str,
    observation: str, observer_username: str,
) -> None:
    try:
        existing = await user_memories.get_facts(chat_id=chat_id, user_id=user_id)
        existing_block = "\n".join(f"- {fact}" for fact in existing) if existing else "(none)"
        prompt = (
            f"Пользователь: @{username}\nИзвестные факты:\n{existing_block}\n\n"
            f"Наблюдение от @{observer_username}: {observation}\n\n"
            f"Что это говорит нам о @{username}? Новые факты (JSON-массив):"
        )
        llm = ChatGroq(model=MEMORY_MODEL, api_key=config.GROQ_API_KEY, temperature=0.2, max_tokens=256)
        result = await llm.ainvoke([SystemMessage(content=EXTRACTION_SYSTEM), HumanMessage(content=prompt)])
        new_facts = _parse_facts(result.content.strip())
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


class MemoryWriter:
    """Fires background fact-extraction and upsert; returns immediately."""

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        response = state.get("response") or ""
        user_message = msg["processed_text"] or msg["raw_text"] or ""

        if msg.get("is_forwarded"):
            return {}

        passive = not response.strip()
        if passive and len(user_message.strip()) < MIN_PASSIVE_LENGTH:
            return {}

        asyncio.create_task(
            extract_and_save(
                chat_id=msg["chat_id"],
                user_id=msg["user_id"],
                username=msg["username"],
                user_message=user_message,
                bot_reply=response,
            )
        )
        return {}
