"""Engagement-mode selection for life posts — talking with the chat, not just about Жора.

Each post is either SOLO (Жора's own story, closing with a question aimed at
the chat) or MEMBER (a real chat member appears in the story via one of
their own stored facts). The mode is picked once per post; member mode
falls back to solo when no eligible member fact exists.
"""

import random

from src import achievements, log
from src.store import user_memories

logger = log.get_logger(__name__)

MEMBER_MODE_CHANCE = 0.5

SOLO = "solo"
MEMBER = "member"

# Facts that would misrepresent or embarrass someone if broadcast to the
# whole chat: counter tallies (insult/hack-attempt stats) read as calling
# the person out, and cross-user attributed facts («по словам @X, ...») are
# hearsay, never confirmed by the person themselves.
UNSAFE_FACT_PREFIXES = ("Оскорблял бота", "Пытался взломать бота", "по словам @")


def is_safe_to_mention(fact: str) -> bool:
    """True when a stored fact is safe to weave into a public life post.

    A mechanical first pass, not a full judgment call: the episode-writer
    prompt separately instructs the model to soften or skip anything that
    reads as too personal, since not every unsafe fact is prefix-detectable.

    Args:
        fact: A stored ``user_memories`` fact string.

    Returns:
        False for counter-tally facts and second-hand cross-user facts;
        True otherwise.
    """
    return not fact.startswith(UNSAFE_FACT_PREFIXES)


async def collect_mentionable_facts() -> list[tuple[str, str]]:
    """Gather (username, fact) candidates across every chat the bot knows about.

    Aggregates across all registered chats (in practice a single friend
    group) rather than scoping to one, since life posts are broadcast
    identically everywhere — see ``bot_memories``' "one life" design.

    Returns:
        All eligible ``(username, fact)`` pairs; may be empty.
    """
    candidates: list[tuple[str, str]] = []
    chat_ids = await achievements.get_all_chat_ids()
    for chat_id in chat_ids:
        members = await achievements.get_chat_members(chat_id)
        if not members:
            continue
        facts_by_uid = await user_memories.get_facts_for_users(
            chat_id=chat_id, user_ids=[user_id for user_id, _ in members]
        )
        usernames = dict(members)
        for user_id, facts in facts_by_uid.items():
            username = usernames.get(user_id)
            if not username:
                continue
            candidates.extend(
                (username, fact) for fact in facts if is_safe_to_mention(fact)
            )
    return candidates


async def choose_mode() -> tuple[str, tuple[str, str] | None]:
    """Pick this episode's engagement mode.

    Fails soft to solo mode on any error — a broken personalization lookup
    must never block a scheduled post.

    Returns:
        ``(MEMBER, (username, fact))`` when member mode is picked and an
        eligible candidate exists; ``(SOLO, None)`` otherwise.
    """
    if random.random() >= MEMBER_MODE_CHANCE:
        return SOLO, None
    try:
        candidates = await collect_mentionable_facts()
    except Exception as err:
        logger.warning("Failed to collect mentionable facts, falling back to solo: %s", err)
        return SOLO, None
    if not candidates:
        return SOLO, None
    return MEMBER, random.choice(candidates)
