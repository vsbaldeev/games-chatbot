"""Emoji-reaction tracking on bot messages.

Requires the bot to be a group admin — Telegram only delivers
``message_reaction`` updates to admins (allowed_updates already includes
Update.ALL_TYPES, see src/app/app.py). Telegram sends the full old/new
reaction lists on every update, never a single delta, so the delta is
computed here and applied to bot_message_feedback (a no-op for any message
that isn't the bot's, or whose 15-minute window has closed — see
src.store.message_feedback).
"""

from telegram import ReactionType, Update
from telegram.ext import ContextTypes

from src import log
from src.store import message_feedback

logger = log.get_logger(__name__)

GROUP_CHAT_TYPES = ("group", "supergroup")


def reaction_key(reaction: ReactionType) -> str:
    """Return a stable string key for one reaction, for counting and diffing.

    Args:
        reaction: A ReactionTypeEmoji, ReactionTypeCustomEmoji, or ReactionTypePaid.

    Returns:
        The bare emoji for a standard reaction, "custom:<id>" for a custom
        emoji reaction, or "paid" for a Telegram Star reaction.
    """
    if reaction.type == "custom_emoji":
        return f"custom:{reaction.custom_emoji_id}"
    if reaction.type == "paid":
        return "paid"
    return reaction.emoji


def diff_reactions(
    old_reaction: list[ReactionType], new_reaction: list[ReactionType],
) -> tuple[list[str], list[str]]:
    """Compute which reaction keys were added and removed between two snapshots.

    Args:
        old_reaction: Reactions before this update.
        new_reaction: Reactions after this update.

    Returns:
        (added_keys, removed_keys) — keys present in only one side.
    """
    old_keys = {reaction_key(reaction) for reaction in old_reaction}
    new_keys = {reaction_key(reaction) for reaction in new_reaction}
    added = [key for key in new_keys if key not in old_keys]
    removed = [key for key in old_keys if key not in new_keys]
    return added, removed


async def handle_message_reaction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Apply an emoji-reaction change to the reacted-to message's feedback row.

    A no-op outside groups, and (via message_feedback's own window guard) a
    no-op for a reaction on a message that isn't a tracked bot message, or
    whose 15-minute feedback window has already closed.
    """
    if update.effective_chat is None or update.effective_chat.type not in GROUP_CHAT_TYPES:
        return
    reaction_update = update.message_reaction
    if reaction_update is None:
        return
    chat_id = reaction_update.chat.id
    message_id = reaction_update.message_id
    added, removed = diff_reactions(reaction_update.old_reaction, reaction_update.new_reaction)
    try:
        for emoji in added:
            await message_feedback.add_reaction(chat_id=chat_id, message_id=message_id, emoji=emoji)
        for emoji in removed:
            await message_feedback.remove_reaction(chat_id=chat_id, message_id=message_id, emoji=emoji)
    except Exception as err:
        logger.warning("Failed to apply reaction change for message %s: %s", message_id, err)
