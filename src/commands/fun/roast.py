"""
"Прожарка" (roast) feature.

Roaster encapsulates LLM-based roast generation and the /roast command handler.
Module-level wrappers preserve the public API that bot.py and handlers.py import.
"""

import random

import numpy as np
from telegram import Update
from telegram.ext import ContextTypes

from src import achievements, config, log
from src.achievements import notify_unlocks
from src.agent import roast_agent
from src.store import embedder, unified_messages
from src.store.roast_store import get_recent_modes, log_roast, pop_roast_target
from src.store.user_memories import get_facts, get_facts_with_embeddings

logger = log.get_logger(__name__)

ROAST_HEADERS = ("💀", "😤", "🎮", "🔥", "💢")

ROAST_ANCHORS = {
    "shame": "провал поражение слабость позор неловкость плохая привычка стыд проигрыш",
    "quirk": "странность привычка особенность манера поведение повторяет всегда никогда",
    "boast": "хвастовство преувеличение самолюбование заявил гордится переоценивает",
}

ANCHOR_CANDIDATE_COUNT = 8
SELECTION_SIZE = 3
EMBARRASSMENT_WEIGHT = 0.3
SOFTMAX_TEMPERATURE = 0.15

CONTRADICTION_MODE = "contradiction"
ROAST_MODES = (*ROAST_ANCHORS, CONTRADICTION_MODE)
CONTRADICTION_FACT_LIMIT = 12

RECENT_MODE_WINDOW = 1

STANDARD_INSTRUCTION = (
    "Обыграй один из этих фактов так, "
    "чтобы засмеялся любой в зале, даже незнакомый с играми и аниме. "
    "Максимум две фразы."
)
CONTRADICTION_INSTRUCTION = (
    "Среди фактов найди самое смешное противоречие или лицемерие "
    "(например, переживает за животных, но ест говядину) и построй прожарку "
    "на этом контрасте. Если явного противоречия нет — высмей самый нелепый факт. "
    "Максимум две фразы."
)
SILENCE_INSTRUCTION = "вообще ничего не пишет в чате. Затроль его за молчание."


def rank_by_anchor(
    facts_with_embeddings: list[tuple[str, np.ndarray]],
    anchor: np.ndarray,
) -> list[tuple[str, np.ndarray, float]]:
    """Rank facts by how embarrassing they are, keeping the strongest candidates.

    Args:
        facts_with_embeddings: Facts paired with their raw embedding vectors.
        anchor: Unit-normalized "embarrassment" anchor embedding.

    Returns:
        Up to ``ANCHOR_CANDIDATE_COUNT`` tuples of ``(fact, unit_embedding,
        anchor_similarity)``, ordered from most to least embarrassing.
    """
    scored = []
    for fact, embedding in facts_with_embeddings:
        norm = np.linalg.norm(embedding)
        unit = embedding / norm if norm > 0 else embedding
        scored.append((fact, unit, float(anchor @ unit)))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:ANCHOR_CANDIDATE_COUNT]


def weighted_choice(items: list, scores: list[float]):
    """Randomly pick one item, softmax-weighting the given scores.

    Higher-scoring items are more likely but never guaranteed, so repeated draws
    over the same candidates yield varied picks — the source of roast variety.

    Args:
        items: Candidate items to choose from.
        scores: Parallel scores; higher means more likely to be chosen.

    Returns:
        One randomly chosen item.
    """
    weights = np.exp((np.array(scores) - np.max(scores)) / SOFTMAX_TEMPERATURE)
    return random.choices(items, weights=weights.tolist(), k=1)[0]


def select_diverse(candidates: list[tuple[str, np.ndarray, float]]) -> list[str]:
    """Pick embarrassing yet topically varied facts, with run-to-run randomness.

    The first fact is drawn softmax-weighted by anchor similarity; each later pick is
    drawn softmax-weighted by ``EMBARRASSMENT_WEIGHT * anchor_similarity minus
    (1 - EMBARRASSMENT_WEIGHT) * redundancy``, where redundancy is the highest
    similarity to an already-chosen fact. The redundancy penalty stops one dense
    cluster (e.g. a single fandom) from filling every slot; the weighted draw keeps
    the chosen set fresh across repeated roasts of the same user.

    Args:
        candidates: Anchor-ranked ``(fact, unit_embedding, anchor_similarity)`` tuples.

    Returns:
        ``SELECTION_SIZE`` facts, biased toward embarrassing and distinct ones.
    """
    chosen = [weighted_choice(candidates, [anchor_sim for _, _, anchor_sim in candidates])]
    while len(chosen) < SELECTION_SIZE and len(chosen) < len(candidates):
        chosen_facts = {fact for fact, _, _ in chosen}
        remaining = [item for item in candidates if item[0] not in chosen_facts]
        scores = [
            EMBARRASSMENT_WEIGHT * anchor_sim
            - (1.0 - EMBARRASSMENT_WEIGHT) * max(float(unit @ picked) for _, picked, _ in chosen)
            for _, unit, anchor_sim in remaining
        ]
        chosen.append(weighted_choice(remaining, scores))
    return [fact for fact, _, _ in chosen]


def pick_roast_facts(
    facts_with_embeddings: list[tuple[str, np.ndarray]],
    anchor_embedding: np.ndarray,
) -> list[str]:
    """Surface embarrassing facts, then spread the final pick across topics.

    Args:
        facts_with_embeddings: Facts paired with their raw embedding vectors.
        anchor_embedding: The "embarrassment" anchor embedding for this roast.

    Returns:
        Up to ``SELECTION_SIZE`` facts to hand the roast model.
    """
    anchor = anchor_embedding / (np.linalg.norm(anchor_embedding) or 1.0)
    candidates = rank_by_anchor(facts_with_embeddings, anchor)
    if len(candidates) <= SELECTION_SIZE:
        return [fact for fact, _, _ in candidates]
    return select_diverse(candidates)


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


async def select_roast_facts(chat_id: int, user_id: int, mode: str) -> list[str]:
    """Select the facts to feed the roast model for the given mode.

    Contradiction mode hands over the recent fact list whole so the model can spot a
    hypocritical pair (e.g. cares about animals yet eats beef); embeddings cannot tell
    a funny stance clash from a consistent one, so no pre-filtering is applied. Anchor
    modes use embedding retrieval plus diverse selection.

    Args:
        chat_id: Telegram chat ID used to look up user facts.
        user_id: Telegram user ID of the roast target.
        mode: The chosen roast mode.

    Returns:
        Facts to hand the roast model (possibly empty if the user has none).
    """
    if mode == CONTRADICTION_MODE:
        facts = await get_facts(chat_id=chat_id, user_id=user_id)
        return facts[:CONTRADICTION_FACT_LIMIT]
    facts_with_embs = await get_facts_with_embeddings(chat_id=chat_id, user_id=user_id)
    if facts_with_embs:
        anchor_emb = np.array(await embedder.embed(ROAST_ANCHORS[mode]))
        return pick_roast_facts(facts_with_embs, anchor_emb)
    return await get_facts(chat_id=chat_id, user_id=user_id)


def build_roast_prompt(mode: str, target_username: str, facts: list[str]) -> str:
    """Assemble the user prompt for the roast model.

    Args:
        mode: The chosen roast mode (selects the instruction line).
        target_username: Username of the roast target (without ``@``).
        facts: Selected facts; empty triggers the silent-member fallback.

    Returns:
        Formatted prompt string for the roast model.
    """
    if not facts:
        return f"@{target_username} {SILENCE_INSTRUCTION}"
    facts_text = "\n".join(f"- {fact}" for fact in facts)
    instruction = CONTRADICTION_INSTRUCTION if mode == CONTRADICTION_MODE else STANDARD_INSTRUCTION
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
        selected = await select_roast_facts(chat_id, user_id, mode)
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
