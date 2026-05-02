"""
GuardNode — third node in the LangGraph pipeline.

Classifies processed_text with llama-prompt-guard-2-86m before the main LLM
is ever invoked.  On MALICIOUS:
  - picks a random refusal from a fixed pool (no LLM call)
  - records/increments a hack-attempt fact in user_memories (async background)
  - sets blocked=True so the graph routes directly to END

On BENIGN passes through unchanged.

Fails open: if the guard API is unavailable the message is allowed through
rather than blocking all chat traffic.
"""

import asyncio
import random
from src import log

from groq import AsyncGroq

from src import config
from src.pipeline.state import BotState
from src.store import user_memories

logger = log.get_logger(__name__)

GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"

GUARD_REFUSALS = [
    "Не выйдет.",
    "Попытка засчитана, но нет.",
    "Креативно, но нет.",
    "Серьёзно? Нет.",
    "Видел и не такое. Нет.",
    "Ни разу не сработает. Нет.",
    "О, классика. Нет.",
    "Продолжай пробовать, я подожду. Нет.",
    "Это называется «попытка взлома». Ответ — нет.",
    "Зачёт за старание, но всё равно нет.",
]


class GuardNode:
    """Runs prompt-injection classification and short-circuits blocked messages."""

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        text = msg.get("processed_text") or msg.get("raw_text") or ""

        if not text.strip():
            return {"blocked": False}

        label = await self.__classify(text)

        if label == "MALICIOUS":
            logger.warning(
                "Guard blocked message from @%s in chat %s", msg["username"], msg["chat_id"]
            )
            asyncio.create_task(self.__record_hack_attempt(msg))
            return {
                "blocked": True,
                "response": random.choice(GUARD_REFUSALS),
            }

        return {"blocked": False}

    async def __classify(self, text: str) -> str:
        try:
            client = AsyncGroq(api_key=config.GROQ_API_KEY)
            result = await client.chat.completions.create(
                model=GUARD_MODEL,
                messages=[{"role": "user", "content": text}],
            )
            return result.choices[0].message.content.strip().upper()
        except Exception as err:
            logger.warning("Guard classification failed, failing open: %s", err)
            return "BENIGN"

    async def __record_hack_attempt(self, msg) -> None:
        try:
            await user_memories.upsert_hack_attempt(
                chat_id=msg["chat_id"],
                user_id=msg["user_id"],
                username=msg["username"],
            )
        except Exception as err:
            logger.warning("Failed to record hack attempt for @%s: %s", msg["username"], err)
