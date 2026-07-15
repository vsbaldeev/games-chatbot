"""Conversation wind-down engine — the unified per-user attention budget.

Replaces the old in-memory insult ladder (``insult_gate``) with one leaky
bucket per (chat, user), persisted in Postgres: every message the bot has to
deal with charges a weight (hostility weighs more than a normal turn), the
score decays with a 30-minute half-life, and the post-charge score maps onto
a wind-down tier — full reply, short in-character brush-off, emoji reaction,
or silence. Any sustained conversation therefore fades out like a person
losing interest, regardless of whether it is hostile, banterous, or genuinely
meaningful.

The engine is pure scoring and tiering; how each tier is presented (which
prompt directive, which emoji pool) stays with the caller.
"""

import time

from src import log
from src.store import engagement

logger = log.get_logger(__name__)

# Weight each classification charges against the attention budget. BOT_INSULT
# stays low enough that two rapid insults both land in the full-reply tier:
# the classifier has false positives, and one mislabel must not cost the user
# their comeback.
SIGNAL_WEIGHTS = {
    "BOT_INSULT": 3.0,
    "BANTER": 2.0,
    "MEANINGLESS": 2.0,
    "MEANINGFUL": 1.0,
}

HALF_LIFE_SECONDS = 30 * 60

FULL_TIER = 1
BRUSH_OFF_TIER = 2
EMOJI_TIER = 3
SILENCE_TIER = 4

# A rapid meaningful run gives ~7 full replies, ~6 brush-offs, ~6 emoji, then
# silence; a rapid insult run (weight 3) gives two comebacks before winding
# down. Paced conversation (one exchange per 10+ minutes) converges below the
# brush-off threshold and never fades.
BRUSH_OFF_THRESHOLD = 7.0
EMOJI_THRESHOLD = 13.0
SILENCE_THRESHOLD = 19.0

# The bot must not initiate jokes at users it is winding down — a joke would
# restart the very conversation the engine is ending.
HUMOR_SUPPRESS_THRESHOLD = BRUSH_OFF_THRESHOLD


def tier_for_score(score: float) -> int:
    """Map a decayed attention score onto a wind-down tier.

    Args:
        score: Post-charge decayed score.

    Returns:
        One of ``FULL_TIER``, ``BRUSH_OFF_TIER``, ``EMOJI_TIER``,
        ``SILENCE_TIER``.
    """
    if score > SILENCE_THRESHOLD:
        return SILENCE_TIER
    if score > EMOJI_THRESHOLD:
        return EMOJI_TIER
    if score > BRUSH_OFF_THRESHOLD:
        return BRUSH_OFF_TIER
    return FULL_TIER


async def register_signal(*, chat_id: int, user_id: int, classification: str) -> int:
    """Charge the user's attention budget and return the wind-down tier.

    Fails open to ``FULL_TIER`` on any store error, so a database hiccup
    degrades to the pre-engine behaviour (always reply) rather than to
    unexplained silence.

    Args:
        chat_id: Chat the message arrived in.
        user_id: Author of the message.
        classification: Filter verdict, a key of ``SIGNAL_WEIGHTS``.

    Returns:
        The tier the bot's reaction should be picked from.
    """
    weight = SIGNAL_WEIGHTS.get(classification, SIGNAL_WEIGHTS["MEANINGFUL"])
    try:
        score = await engagement.add_signal(
            chat_id=chat_id, user_id=user_id, weight=weight,
            now=time.time(), half_life_seconds=HALF_LIFE_SECONDS,
        )
    except Exception as error:
        logger.warning(
            "Engagement: failed to charge chat=%s user=%s — failing open: %s",
            chat_id, user_id, error,
        )
        return FULL_TIER
    tier = tier_for_score(score)
    logger.info(
        "Engagement: chat=%s user=%s signal=%s score=%.1f tier=%s",
        chat_id, user_id, classification, score, tier,
    )
    return tier


async def is_wound_down(*, chat_id: int, user_id: int) -> bool:
    """Check whether a user is past the point of bot-initiated attention.

    Read-only: peeks the decayed score without charging the budget. Fails
    open to ``False`` — a missed suppression is a harmless extra joke.

    Args:
        chat_id: Chat the user is in.
        user_id: User to check.

    Returns:
        True when the user's score exceeds ``HUMOR_SUPPRESS_THRESHOLD``.
    """
    try:
        score = await engagement.peek_score(
            chat_id=chat_id, user_id=user_id,
            now=time.time(), half_life_seconds=HALF_LIFE_SECONDS,
        )
    except Exception as error:
        logger.warning(
            "Engagement: peek failed for chat=%s user=%s — allowing: %s",
            chat_id, user_id, error,
        )
        return False
    return score > HUMOR_SUPPRESS_THRESHOLD
