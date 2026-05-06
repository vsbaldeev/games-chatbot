"""
MemoryWriter — fires fact extraction and persist in the background.

Runs in two modes:
  - Active (bot replied): fed the full exchange (user message + bot reply).
  - Passive (no reply): fed the user message alone — the bot "overheard" it.

Extraction uses llama-3.1-8b-instant and never blocks the pipeline.
"""

import asyncio
import json
from src import log

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.pipeline.state import BotState
from src.store import user_memories

logger = log.get_logger(__name__)

MEMORY_MODEL = "llama-3.1-8b-instant"
MAX_NEW_FACTS = 3
MIN_PASSIVE_LENGTH = 20

EXTRACTION_SYSTEM = (
    "You extract concise facts about a person from a single chat exchange. "
    "Return a JSON array of short English strings (max 15 words each). "
    "Only include facts that are new or update existing ones. "
    "Return [] if nothing new was learned. No explanation, no markdown — raw JSON only."
)


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
            self.__extract_and_save(
                chat_id=msg["chat_id"],
                user_id=msg["user_id"],
                username=msg["username"],
                user_message=user_message,
                bot_reply=response,
            )
        )
        return {}

    async def __call_extraction_llm(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        user_message: str,
        bot_reply: str,
    ) -> list[str]:
        existing = await user_memories.get_facts(chat_id=chat_id, user_id=user_id)
        existing_block = "\n".join(f"- {fact}" for fact in existing) if existing else "(none)"
        if bot_reply:
            exchange = f"Exchange:\n@{username}: {user_message}\nBot: {bot_reply}"
        else:
            exchange = f"Message (bot was not addressed):\n@{username}: {user_message}"
        prompt = (
            f"User: @{username}\n"
            f"Existing facts:\n{existing_block}\n\n"
            f"{exchange}\n\n"
            f"New facts to add (JSON array):"
        )
        llm = ChatGroq(
            model=MEMORY_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.2,
            max_tokens=256,
        )
        result = await llm.ainvoke([
            SystemMessage(content=EXTRACTION_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return self.__parse_facts(result.content.strip())

    async def __extract_and_save(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        user_message: str,
        bot_reply: str,
    ) -> None:
        try:
            new_facts = await self.__call_extraction_llm(
                chat_id, user_id, username, user_message, bot_reply
            )
            if new_facts:
                await user_memories.upsert_facts(
                    chat_id=chat_id,
                    user_id=user_id,
                    username=username,
                    facts=new_facts[:MAX_NEW_FACTS],
                )
                logger.debug("Saved %d facts for @%s in chat %s", len(new_facts), username, chat_id)
        except Exception as err:
            logger.warning("Memory extraction failed for @%s: %s", username, err)

    @staticmethod
    def __parse_facts(raw: str) -> list[str]:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return []
