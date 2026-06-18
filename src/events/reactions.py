"""Reaction handler — emoji stat tracking and crediting."""

from src import log
from telegram import Update
from telegram.ext import ContextTypes

from src import achievements
from src.achievements import notify_unlocks
from src.store.roast_store import is_roast_message, record_reaction

logger = log.get_logger(__name__)


async def __credit_reaction_stats(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    author_id: int,
    author_username: str,
    added_emojis: set[str],
) -> None:
    laugh_emojis = {"😁", "🤣"}
    heart_emojis = {"❤", "🥰", "😍", "💘", "❤️‍\U0001f525"}
    fire_emojis = {"🔥"}
    thumb_emojis = {"👍"}

    stat_map = [
        (laugh_emojis, "laugh_reactions"),
        (heart_emojis, "heart_reactions"),
        (fire_emojis, "fire_reactions"),
        (thumb_emojis, "thumbsup_reactions"),
    ]
    credited_any = False
    for emoji_set, stat_name in stat_map:
        if added_emojis & emoji_set:
            await achievements.increment_stat(author_id, chat_id, author_username, stat_name)
            credited_any = True

    if credited_any:
        await notify_unlocks(context, chat_id, author_id, author_username)


async def handle_reaction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    reaction = update.message_reaction
    if not reaction:
        return

    chat_id = reaction.chat.id
    old_emojis = {react_item.emoji for react_item in reaction.old_reaction if hasattr(react_item, "emoji")}
    new_emojis = {react_item.emoji for react_item in reaction.new_reaction if hasattr(react_item, "emoji")}
    added_emojis = new_emojis - old_emojis
    removed_emojis = old_emojis - new_emojis

    if (added_emojis or removed_emojis) and await is_roast_message(reaction.message_id, chat_id):
        for emoji in added_emojis:
            await record_reaction(reaction.message_id, chat_id, emoji, +1)
        for emoji in removed_emojis:
            await record_reaction(reaction.message_id, chat_id, emoji, -1)

    if not added_emojis:
        return
    author = await achievements.get_message_author(chat_id, reaction.message_id)
    if not author:
        logger.debug(
            "No author cached for message %s in chat %s — skipping reaction",
            reaction.message_id, chat_id,
        )
        return
    author_id, author_username = author
    if reaction.user and reaction.user.id == author_id:
        return
    logger.info(
        "Reaction %s on message %s in chat %s credited to %s",
        added_emojis, reaction.message_id, chat_id, author_username,
    )

    await __credit_reaction_stats(context, chat_id, author_id, author_username, added_emojis)
