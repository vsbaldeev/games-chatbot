"""
MemoryWriter — fires fact extraction and upsert in the background.

Runs in two modes:
  - Active (bot replied): fed the full exchange (user message + bot reply).
  - Passive (no reply): fed the user message alone — the bot "overheard" it.

The module-level extract_and_save is also called directly from the router
for plain text messages that don't trigger a bot response.
Cross-user facts are extracted automatically when @mentions are present.
"""

import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import achievements, config, log
from src.pipeline.state import BotState
from src.store import user_memories

logger = log.get_logger(__name__)

MEMORY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_NEW_FACTS = 3
MIN_PASSIVE_LENGTH = 20

MENTION_RE = re.compile(r"@(\w+)", re.UNICODE)

EXTRACTION_SYSTEM = (
    "You extract concise facts about a person from a single chat exchange. "
    "Return a JSON array of short strings in the same language the user wrote in (max 15 words each). "
    "Only include facts that are new or update existing ones. "
    "Return [] if nothing new was learned. No explanation, no markdown — raw JSON only."
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


async def extract_and_save(
    *, chat_id: int, user_id: int, username: str, user_message: str, bot_reply: str = ""
) -> None:
    """Extract facts about the sender and any @mentioned users. Safe to use with create_task."""
    try:
        existing = await user_memories.get_facts(chat_id=chat_id, user_id=user_id)
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
        new_facts = _parse_facts(result.content.strip())
        if new_facts:
            await user_memories.upsert_facts(
                chat_id=chat_id, user_id=user_id, username=username,
                facts=new_facts[:MAX_NEW_FACTS],
            )
            logger.debug("Saved %d facts for @%s in chat %s", len(new_facts), username, chat_id)
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
            f"User: @{username}\nExisting facts:\n{existing_block}\n\n"
            f"Observation by @{observer_username}: {observation}\n\n"
            f"What does this tell us about @{username}? New facts (JSON array):"
        )
        llm = ChatGroq(model=MEMORY_MODEL, api_key=config.GROQ_API_KEY, temperature=0.2, max_tokens=256)
        result = await llm.ainvoke([SystemMessage(content=EXTRACTION_SYSTEM), HumanMessage(content=prompt)])
        new_facts = _parse_facts(result.content.strip())
        if new_facts:
            await user_memories.upsert_facts(
                chat_id=chat_id, user_id=user_id, username=username,
                facts=new_facts[:MAX_NEW_FACTS],
            )
            logger.debug(
                "Saved %d cross-user facts for @%s (observed by @%s)",
                len(new_facts), username, observer_username,
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
