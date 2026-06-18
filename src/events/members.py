"""Member tracking handlers — registration and seeding."""

from src import log
from telegram import Update
from telegram.ext import ContextTypes

from src import achievements

logger = log.get_logger(__name__)


def get_username(update: Update) -> str:
    user = update.effective_user
    return user.username or user.first_name or f"user_{user.id}"


async def track_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_chat:
        return
    user = update.effective_user
    username = user.username or user.first_name or f"user_{user.id}"
    await achievements.register_member(update.effective_chat.id, user.id, username, is_bot=user.is_bot)

    if update.message and not user.is_bot:
        await achievements.set_message_author(
            update.effective_chat.id, update.message.message_id, user.id, username
        )


async def handle_new_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.new_chat_members:
        return
    chat_id = update.effective_chat.id
    for user in update.message.new_chat_members:
        username = user.username or user.first_name or f"user_{user.id}"
        await achievements.register_member(chat_id, user.id, username, is_bot=user.is_bot)


async def handle_bot_added_to_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Seed chat_members with current admins when the bot is added to a group."""
    if not update.my_chat_member:
        return
    new_status = update.my_chat_member.new_chat_member.status
    if new_status not in ("member", "administrator"):
        return
    chat_id = update.effective_chat.id
    try:
        admins = await context.bot.get_chat_administrators(chat_id)
        for admin in admins:
            username = admin.user.username or admin.user.first_name or f"user_{admin.user.id}"
            await achievements.register_member(chat_id, admin.user.id, username, is_bot=admin.user.is_bot)
        logger.info("Seeded %d admins for chat %s on bot join", len(admins), chat_id)
    except Exception as error:
        logger.warning("Failed to seed admins for chat %s: %s", chat_id, error)
