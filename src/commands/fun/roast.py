"""
"Прожарка" (roast) feature.

Roaster encapsulates LLM-based roast generation and the /roast command handler.
Module-level wrappers preserve the public API that bot.py and handlers.py import.
"""

import random

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import Update
from telegram.ext import ContextTypes

from src import achievements, config, log
from src.achievements import notify_unlocks
from src.agent import apply_language_correction
from src.store.user_memories import get_facts

logger = log.get_logger(__name__)

ROAST_MODEL = "llama-3.3-70b-versatile"

ROAST_HEADERS = ("💀", "😤", "🎮", "🔥", "💢")

SYSTEM_PROMPT = (
    "Ты — токсичный школьник-геймер. "
    "Если человек долго молчал — высмей "
    "Отвечай коротко и токсично. Только русский язык. Не объясняй шутку."
    "Можно материться."
)


class Roaster:
    """Generates LLM-powered roasts and handles the /roast Telegram command."""

    async def generate(self, chat_id: int, user_id: int, target_username: str) -> tuple[str, str]:
        llm = ChatGroq(model=ROAST_MODEL, api_key=config.GROQ_API_KEY, temperature=0.65, max_tokens=120)
        facts = await get_facts(chat_id=chat_id, user_id=user_id)
        if facts:
            facts_text = "\n".join(f"- {fact}" for fact in facts)
            user_prompt = (
                f"Факты о @{target_username}:\n{facts_text}\n\n"
                f"Затроль @{target_username} на основе этих фактов."
            )
        else:
            user_prompt = f"@{target_username} вообще ничего не пишет в чате. Затроль его за молчание."
        header = random.choice(ROAST_HEADERS)
        response = await self.__invoke(llm, SYSTEM_PROMPT, user_prompt)
        return header, response.content

    async def __invoke(self, llm: ChatGroq, system_prompt: str, user_prompt: str):
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = await llm.ainvoke(messages)
        return await apply_language_correction(llm, response, messages)

    async def cmd_roast(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        members = await achievements.get_chat_members(chat_id)
        if not members:
            await update.message.reply_text(
                "В базе нет участников. Пусть сначала кто-нибудь напишет в чат."
            )
            return
        target_id, target_username = random.choice(members)
        await update.message.chat.send_action("typing")
        try:
            header, roast_text = await self.generate(chat_id, target_id, target_username)
            await update.message.reply_text(f"{header} #прожарка @{target_username}\n\n{roast_text}")
            await achievements.increment_stat(target_id, chat_id, target_username, "roasted_count")
            await notify_unlocks(context, chat_id, target_id, target_username)
        except Exception as error:
            logger.error("Roast failed for %s in chat %s: %s", target_username, chat_id, error)
            await update.message.reply_text(
                "Прожарка не задалась. Groq на перекуре — попробуй позже."
            )


# ---------------------------------------------------------------------------
# Module-level singleton + backward-compatible wrappers
# ---------------------------------------------------------------------------

roaster = Roaster()


async def generate_roast_text(chat_id: int, user_id: int, target_username: str) -> tuple[str, str]:
    return await roaster.generate(chat_id, user_id, target_username)


async def cmd_roast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await roaster.cmd_roast(update, context)
