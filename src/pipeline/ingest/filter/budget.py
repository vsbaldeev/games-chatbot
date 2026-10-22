"""
charge_budget — filter-stage node that charges the sender's attention budget.

Every verdict charges the sender's persistent attention budget (see
``engagement_gate``), and the post-charge wind-down tier decides the shape of
the reaction: a full reply, a short LLM-generated in-character brush-off
(``wind_down`` state flag), a bored emoji reaction, or silence — so any
sustained conversation fades out like a person losing interest. Counter-insults
that reply to the bot's own message never earn a fresh full comeback (that is
what fuels roast-battle loops).

Emoji reactions fire via ``asyncio.create_task`` and do not block the pipeline.
"""

import asyncio
import random

from telegram import ReactionTypeEmoji

from src import config, log
from src.config.prompts import GROUP_PROFILE_COOLDOWN_REPLIES, GROUP_PROFILE_COOLDOWN_SECONDS
from src.pipeline.ingest import engagement_gate
from src.pipeline.state import BotState, ContentKind, Verdict
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

REACTION_POOL = ["👍", "❤", "🔥", "😁", "👀", "😎", "💯", "🤣", "⚡", "🫡", "🎉"]

# Bored acknowledgement for users the engagement gate has wound down to the
# emoji tier — must stay within Telegram's fixed set of allowed reaction emoji.
DISMISSIVE_REACTIONS = ["🥱", "😴", "🗿", "🤨"]

# Per-chat cooldown between accepted group-profile requests (see
# GROUP_PROFILE_COOLDOWN_SECONDS) — one run costs an LLM call plus a store
# query per chat member, well above an ordinary reply.
group_profile_cooldown_gate = TtlGate(GROUP_PROFILE_COOLDOWN_SECONDS)


def replies_to_bot(msg: dict) -> bool:
    """Check whether the incoming message replies to one of the bot's own messages.

    Uses the ``replied_to_fallback`` synthesized from the Telegram update
    (always present for replies, no store lookup needed).

    Args:
        msg: IncomingMessage dict of the message being classified.

    Returns:
        True when the replied-to message was authored by the bot.
    """
    return (msg.get("replied_to_fallback") or {}).get("user_id") == config.BOT_ID


async def send_reaction(state: BotState, pool: list[str] = REACTION_POOL) -> None:
    """React to the message with a random emoji from the given pool.

    Args:
        state: Current pipeline state.
        pool: Emoji to pick from; defaults to the friendly acknowledgement
            pool, with ``DISMISSIVE_REACTIONS`` used for insult barrages.
    """
    try:
        bot = state["context_types"].bot
        msg = state["incoming"]
        emoji = random.choice(pool)
        await bot.set_message_reaction(
            chat_id=msg["chat_id"],
            message_id=msg["message_id"],
            reaction=[ReactionTypeEmoji(emoji=emoji)],
        )
    except Exception as err:  # noqa: BLE001 — a reaction is best-effort, never fatal
        logger.warning("Reaction failed for message %s: %s", state["incoming"]["message_id"], err)


def build_reply_flags(state: BotState, classification: str, tier: int) -> dict:
    """Assemble the state flags for tiers that answer with text.

    A BANTER verdict and any brush-off-tier reply set ``wind_down`` (one
    short conversation-closing phrase, worker skipped). A BOT_INSULT gets
    the full comeback only at the full tier and only when the insult is
    not a reply to the bot's own message — a mirrored counter-insult in a
    running thread would just fuel the loop. A MEME_REQUEST at the full
    tier sets ``meme_request``, whose reply is the image alone; at the
    brush-off tier it falls into the ``wind_down`` refusal.
    GROUP_PROFILE_REQUEST behaves the same way via ``group_profile_request``,
    except a request inside the per-chat cooldown window returns a canned
    refusal with ``should_respond: False`` instead — skipping the LLM call
    entirely rather than winding down the user's own attention budget.

    Args:
        state: Current pipeline state.
        classification: Filter verdict.
        tier: Wind-down tier returned by the engagement gate.

    Returns:
        State update dict, normally with ``should_respond: True``.
    """
    update: dict = {"should_respond": True}
    if classification == Verdict.BOT_INSULT:
        update["is_bot_insult"] = True
        if tier != engagement_gate.FULL_TIER or replies_to_bot(state["incoming"]):
            update["wind_down"] = True
    elif classification == Verdict.MEME_REQUEST and tier == engagement_gate.FULL_TIER:
        update["meme_request"] = True
    elif classification == Verdict.GROUP_PROFILE_REQUEST and tier == engagement_gate.FULL_TIER:
        chat_id = state["incoming"]["chat_id"]
        if group_profile_cooldown_gate.seen(chat_id):
            return {
                "should_respond": False,
                "response": random.choice(GROUP_PROFILE_COOLDOWN_REPLIES),
                "content_kind": ContentKind.CAP_NOTICE,
            }
        update["group_profile_request"] = True
    elif classification == Verdict.BANTER or tier != engagement_gate.FULL_TIER:
        update["wind_down"] = True
    logger.debug(
        "Filter[budget]: %s at tier %s, message %s — replying%s",
        classification, tier, state["incoming"]["message_id"],
        " (wind-down)" if update.get("wind_down") else "",
    )
    return update


async def charge_budget(state: BotState) -> dict:
    """Charge the sender's attention budget and apply the wind-down tier.

    Every verdict spends budget (hostility and banter weigh more than a
    meaningful turn), and the post-charge tier decides the shape of the
    reaction.

    Args:
        state: Current pipeline state; ``filter_verdict`` is a key of
            ``engagement_gate.SIGNAL_WEIGHTS``.

    Returns:
        State update dict.
    """
    msg = state["incoming"]
    classification = state["filter_verdict"]
    tier = await engagement_gate.register_signal(
        chat_id=msg["chat_id"], user_id=msg["user_id"], classification=classification,
    )
    update = apply_tier(state, classification, tier)
    return {"filter_verdict": classification, "engagement_tier": tier, **update}


def apply_tier(state: BotState, classification: Verdict, tier: int) -> dict:
    """Map (classification, tier) onto a reply, an emoji reaction or silence.

    Args:
        state: Current pipeline state.
        classification: Filter verdict.
        tier: Wind-down tier returned by the engagement gate.

    Returns:
        State update dict.
    """
    message_id = state["incoming"]["message_id"]
    if tier == engagement_gate.SILENCE_TIER:
        logger.debug(
            "Filter[budget]: %s from wound-down user, message %s — silence",
            classification, message_id,
        )
        return {"should_respond": False, "drop_reason": "wound_down"}
    if classification == Verdict.MEANINGLESS:
        pool = DISMISSIVE_REACTIONS if tier == engagement_gate.EMOJI_TIER else REACTION_POOL
        logger.debug("Filter[budget]: Dropping meaningless message %s", message_id)
        asyncio.create_task(send_reaction(state, pool))
        return {"should_respond": False, "drop_reason": "meaningless"}
    replies_with_text = tier == engagement_gate.FULL_TIER or (
        tier == engagement_gate.BRUSH_OFF_TIER and classification != Verdict.BANTER
    )
    if not replies_with_text:
        logger.debug(
            "Filter[budget]: %s at tier %s, message %s — emoji reaction only",
            classification, tier, message_id,
        )
        asyncio.create_task(send_reaction(state, DISMISSIVE_REACTIONS))
        return {"should_respond": False, "drop_reason": "wind_down_reaction"}
    return build_reply_flags(state, classification, tier)
