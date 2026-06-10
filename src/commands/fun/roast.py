"""
"Прожарка" (roast) feature.

Roaster encapsulates LLM-based roast generation and the /roast command handler.
Module-level wrappers preserve the public API that bot.py and handlers.py import.
"""

import random

from telegram import Update
from telegram.ext import ContextTypes

from src import achievements, config, log
from src.achievements import notify_unlocks
from src.agent import roast_agent
from src.store import unified_messages
from src.store.roast_store import get_recent_modes, log_roast, pop_roast_target
from src.store.user_memories import get_facts

logger = log.get_logger(__name__)

ROAST_HEADERS = ("💀", "😤", "🎮", "🔥", "💢")

CONTRADICTION_MODE = "contradiction"

# Each mode is an angle hint appended to the fact list. The model receives every
# stored fact and chooses the funniest one itself — there is no pre-filtering, so a
# punchline fact can never be hidden behind a retrieval window. Brevity and the
# one-punch structure live in ROAST_SYSTEM_PROMPT; each line here only picks the angle.
ROAST_MODE_INSTRUCTIONS = {
    "shame": "Зацепись за факт, где он опозорился или сглупил, и врежь по нему.",
    CONTRADICTION_MODE: (
        "Найди ОДНО настоящее противоречие между его фактами (например, кайфует от мотоцикла, "
        "но экономит на такси) и врежь по нему. Не сваливай несколько противоречий в кучу. "
        "Если явного противоречия нет — высмей самый нелепый факт."
    ),
}

ROAST_MODES = tuple(ROAST_MODE_INSTRUCTIONS)

RECENT_MODE_WINDOW = 1

SILENCE_INSTRUCTION = "вообще ничего не пишет в чате. Затроль его за молчание."


def pick_roast_mode(recent_modes: list[str]) -> str:
    """Randomly choose a roast angle, avoiding recently used ones.

    Excluding the modes used in this user's latest roasts keeps the angle fresh
    between back-to-back roasts. If exclusion would leave nothing (few past
    roasts, or all modes recently used), the full mode set is restored.

    Args:
        recent_modes: Anchor keys from this user's most recent roasts.

    Returns:
        One of the embarrassment anchor keys or ``CONTRADICTION_MODE``.
    """
    available = [mode for mode in ROAST_MODES if mode not in recent_modes]
    return random.choice(available or list(ROAST_MODES))


async def select_roast_facts(chat_id: int, user_id: int) -> list[str]:
    """Fetch every stored fact for the roast target, newest first.

    All facts are handed to the roast model so it can choose the funniest one itself.
    The pool is already capped (``MAX_FACTS_PER_USER``) and each fact is a short
    sentence, so the whole list fits comfortably in the model's context. Selecting
    here would risk hiding the very fact that makes the joke land.

    Args:
        chat_id: Telegram chat ID used to look up user facts.
        user_id: Telegram user ID of the roast target.

    Returns:
        Facts to hand the roast model (possibly empty if the user has none).
    """
    return await get_facts(chat_id=chat_id, user_id=user_id)


def build_roast_prompt(mode: str, target_username: str, facts: list[str]) -> str:
    """Assemble the user prompt for the roast model.

    Args:
        mode: The chosen roast mode (selects the angle instruction line).
        target_username: Username of the roast target (without ``@``).
        facts: The target's facts; empty triggers the silent-member fallback.

    Returns:
        Formatted prompt string for the roast model.
    """
    if not facts:
        return f"@{target_username} {SILENCE_INSTRUCTION}"
    facts_text = "\n".join(f"- {fact}" for fact in facts)
    instruction = ROAST_MODE_INSTRUCTIONS[mode]
    return f"Факты о @{target_username}:\n{facts_text}\n\n{instruction}"


class Roaster:
    """Generates LLM-powered roasts and handles the /roast Telegram command."""

    async def generate(self, chat_id: int, user_id: int, target_username: str) -> tuple[str, str, str]:
        """Generate a roast for the target user.

        Args:
            chat_id: Telegram chat ID used to look up user facts.
            user_id: Telegram user ID of the roast target.
            target_username: Username of the roast target (without ``@``).

        Returns:
            Tuple of ``(header_emoji, roast_text, mode)``, where ``mode`` is an
            embarrassment anchor key or ``CONTRADICTION_MODE``.
        """
        recent_modes = await get_recent_modes(chat_id, user_id, RECENT_MODE_WINDOW)
        mode = pick_roast_mode(recent_modes)
        selected = await select_roast_facts(chat_id, user_id)
        user_prompt = build_roast_prompt(mode, target_username, selected)
        header = random.choice(ROAST_HEADERS)
        roast_text = await roast_agent.invoke_roast(user_prompt)
        return header, roast_text, mode

    async def cmd_roast(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        chat_id = update.effective_chat.id
        members = await achievements.get_chat_members(chat_id)
        if not members:
            await update.message.reply_text(
                "В базе нет участников. Пусть сначала кто-нибудь напишет в чат."
            )
            return
        target_id, target_username = await pop_roast_target(chat_id, members)
        await update.message.chat.send_action("typing")
        try:
            header, roast_text, anchor_key = await self.generate(chat_id, target_id, target_username)
            full_text = f"{header} #прожарка @{target_username}\n\n{roast_text}"
            sent = await update.message.reply_text(full_text)
            await log_roast(message_id=sent.message_id, chat_id=chat_id, target_user_id=target_id, anchor_key=anchor_key)
            await unified_messages.insert(
                chat_id=chat_id,
                message_id=sent.message_id,
                user_id=context.bot.id,
                username=config.BOT_USERNAME,
                content=full_text,
                media_type="text",
                reply_to_msg_id=update.message.message_id,
            )
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


async def generate_roast_text(chat_id: int, user_id: int, target_username: str) -> tuple[str, str, str]:
    return await roaster.generate(chat_id, user_id, target_username)


async def cmd_roast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await roaster.cmd_roast(update, context)
