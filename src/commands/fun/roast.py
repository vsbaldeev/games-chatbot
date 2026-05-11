"""
"Прожарка" (roast) feature.

Roaster encapsulates LLM-based roast generation and the /roast command handler.
Module-level wrappers preserve the public API that bot.py and handlers.py import.
"""

import itertools
import random

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import Update
from telegram.ext import ContextTypes

from src import achievements, config, log
from src.achievements import notify_unlocks
from src.agent import apply_language_correction
from src.store import embedder, unified_messages
from src.store.user_memories import get_facts, get_facts_with_embeddings

logger = log.get_logger(__name__)

ROAST_MODEL = "llama-3.3-70b-versatile"

ROAST_HEADERS = ("💀", "😤", "🎮", "🔥", "💢")

SYSTEM_PROMPT = (
    "Ты — острый на язык завсегдатай русского игрового чата. "
    "Тебе дают факты о человеке — уже отобранные, самые неловкие. "
    "Коротко установи контекст, потом одна неожиданная фраза, которая бьёт точно в цель. "
    "Чем конкретнее — тем смешнее. Не объясняй шутку. Только русский. Мат допустим."
)

ROAST_ANCHOR = "провал поражение слабость позор неловкость плохая привычка стыд проигрыш"

ANCHOR_CANDIDATE_COUNT = 8
CLUSTER_SIZE = 3


def pick_roast_cluster(
    facts_with_embeddings: list[tuple[str, np.ndarray]],
    anchor_embedding: np.ndarray,
) -> list[str]:
    """Hybrid selection: anchor retrieval to surface embarrassing facts, then tightest sub-cluster."""
    anchor = anchor_embedding / (np.linalg.norm(anchor_embedding) or 1.0)
    normalized = []
    for fact, emb in facts_with_embeddings:
        norm = np.linalg.norm(emb)
        normalized.append((fact, emb / norm if norm > 0 else emb))

    candidates = sorted(normalized, key=lambda pair: float(anchor @ pair[1]), reverse=True)
    candidates = candidates[:ANCHOR_CANDIDATE_COUNT]

    if len(candidates) <= CLUSTER_SIZE:
        return [fact for fact, _ in candidates]

    best_indices: tuple | None = None
    best_score = -1.0
    for combo in itertools.combinations(range(len(candidates)), CLUSTER_SIZE):
        vecs = [candidates[idx][1] for idx in combo]
        pair_sims = [
            float(vecs[a] @ vecs[b])
            for a in range(len(vecs))
            for b in range(a + 1, len(vecs))
        ]
        score = sum(pair_sims) / len(pair_sims)
        if score > best_score:
            best_score = score
            best_indices = combo

    return [candidates[idx][0] for idx in best_indices]


class Roaster:
    """Generates LLM-powered roasts and handles the /roast Telegram command."""

    async def generate(self, chat_id: int, user_id: int, target_username: str) -> tuple[str, str]:
        llm = ChatGroq(model=ROAST_MODEL, api_key=config.GROQ_API_KEY, temperature=0.5, top_p=0.9, max_tokens=100)
        facts_with_embs = await get_facts_with_embeddings(chat_id=chat_id, user_id=user_id)
        if facts_with_embs:
            anchor_emb = np.array(await embedder.embed(ROAST_ANCHOR))
            selected = pick_roast_cluster(facts_with_embs, anchor_emb)
        else:
            selected = await get_facts(chat_id=chat_id, user_id=user_id)
        if selected:
            facts_text = "\n".join(f"- {fact}" for fact in selected)
            user_prompt = (
                f"Факты о @{target_username}:\n{facts_text}\n\n"
                f"Сделай прожарку на основе этих фактов. Максимум две фразы."
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
        if not update.message:
            return
        chat_id = update.effective_chat.id
        members = [
            member for member in await achievements.get_chat_members(chat_id)
            if member[0] != context.bot.id
        ]
        if not members:
            await update.message.reply_text(
                "В базе нет участников. Пусть сначала кто-нибудь напишет в чат."
            )
            return
        target_id, target_username = random.choice(members)
        await update.message.chat.send_action("typing")
        try:
            header, roast_text = await self.generate(chat_id, target_id, target_username)
            full_text = f"{header} #прожарка @{target_username}\n\n{roast_text}"
            sent = await update.message.reply_text(full_text)
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


async def generate_roast_text(chat_id: int, user_id: int, target_username: str) -> tuple[str, str]:
    return await roaster.generate(chat_id, user_id, target_username)


async def cmd_roast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await roaster.cmd_roast(update, context)
