"""
"Прожарка" (roast) feature.

Roaster encapsulates LLM-based roast generation and the /roast command handler.
Module-level wrappers preserve the public API that bot.py and handlers.py import.
"""

import random
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import Update
from telegram.ext import ContextTypes

from src import achievements, config, log
from src.achievements import notify_unlocks
from src.agent import apply_language_correction
from src.store import unified_messages

logger = log.get_logger(__name__)

ROAST_MODEL = "llama-3.3-70b-versatile"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027FF\U0001FA00-\U0001FAFF]",
    re.UNICODE,
)


class Roaster:
    """Generates LLM-powered roasts and handles the /roast Telegram command."""

    def __build_roast_prompts(
        self, target_username: str, history_text: str, is_supportive: bool
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for the LLM call."""
        if is_supportive:
            style_instruction = (
                f"Напиши искреннее тёплое поддерживающее сообщение для @{target_username} — "
                f"как лучший друг, который реально верит в него. Без сарказма, с душой. До 2 предложений."
            )
        else:
            style_instruction = (
                f"Напиши жёсткий саркастический роаст на @{target_username} в стиле стендап-комика. "
                f"Максимум 2 предложения. Злой юмор, чёрный сарказм, смешно и больно. "
                f"Обязательно упомяни @{target_username} в тексте."
            )

        if history_text:
            system_prompt = (
                "Ты стендап-комик в группе друзей-геймеров. "
                "Пишешь короткие роасты — строго до 2 предложений. "
                "Можно использовать мат и крепкие выражения. "
                "Только русский язык."
            )
            user_prompt = (
                f"Последние сообщения @{target_username} в чате:\n{history_text}\n\n{style_instruction}"
            )
        else:
            system_prompt = (
                "Ты дружелюбный бот в группе друзей-геймеров. "
                "Пишешь тёплые, искренние сообщения — строго до 2 предложений. "
                "Только русский язык."
            )
            user_prompt = (
                f"@{target_username} ещё не написал в чате ни слова. "
                f"Напиши ему тёплое, дружелюбное сообщение от лица чата — "
                f"позови поучаствовать в общении, скажи что рады его видеть. "
                f"Обязательно упомяни @{target_username}. До 2 предложений."
            )
        return system_prompt, user_prompt

    async def generate(self, chat_id: int, user_id: int, target_username: str) -> str:
        """Generate and return a roast string for the given user."""
        llm = ChatGroq(
            model=ROAST_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.95,
            max_tokens=180,
        )
        history_text = await self.__get_user_history_text(chat_id, target_username)
        is_supportive = random.random() < 0.1
        system_prompt, user_prompt = self.__build_roast_prompts(
            target_username, history_text, is_supportive
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = await llm.ainvoke(messages)
        response = await apply_language_correction(llm, response, messages)
        return response.content

    async def cmd_roast(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /roast command — pick a random member and roast them."""
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
            roast_text = await self.generate(chat_id, target_id, target_username)
            await update.message.reply_text(
                f"🔥 Прожарка @{target_username}:\n\n{roast_text}"
            )
            await achievements.increment_stat(target_id, chat_id, target_username, "roasted_count")
            await notify_unlocks(context, chat_id, target_id, target_username)
        except Exception as error:
            logger.error("Roast failed for %s in chat %s: %s", target_username, chat_id, error)
            await update.message.reply_text(
                "Прожарка не задалась. Groq на перекуре — попробуй позже."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def __get_user_history_text(self, chat_id: int, username: str) -> str:
        """Return the last 40 meaningful messages from the given user as a newline-joined string."""
        messages = await unified_messages.get_user_messages(chat_id=chat_id, username=username, limit=40)
        meaningful = [msg for msg in messages if self.__is_meaningful(msg)]
        return "\n".join(meaningful)

    @staticmethod
    def __is_meaningful(text: str) -> bool:
        """Return True if the text contains actual words beyond links and emojis."""
        stripped = URL_RE.sub("", text)
        stripped = EMOJI_RE.sub("", stripped).strip()
        return bool(stripped)


# ---------------------------------------------------------------------------
# Module-level singleton + backward-compatible wrappers
# ---------------------------------------------------------------------------

roaster = Roaster()


async def generate_roast_text(chat_id: int, user_id: int, target_username: str) -> str:
    return await roaster.generate(chat_id, user_id, target_username)


async def cmd_roast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await roaster.cmd_roast(update, context)
