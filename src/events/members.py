"""Member tracking handlers — registration and seeding."""

from telegram import Update
from telegram.ext import ContextTypes

from src import achievements


def get_username(update: Update) -> str:
    user = update.effective_user
    return user.username or user.first_name or f"user_{user.id}"


async def register_sender_as_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_chat:
        return
    user = update.effective_user
    username = user.username or user.first_name or f"user_{user.id}"
    await achievements.register_member(update.effective_chat.id, user.id, username, is_bot=user.is_bot)

    if update.message and not user.is_bot:
        await achievements.set_message_author(
            update.effective_chat.id, update.message.message_id, user.id, username
        )


async def register_users_from_join_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.new_chat_members:
        return
    chat_id = update.effective_chat.id
    for user in update.message.new_chat_members:
        username = user.username or user.first_name or f"user_{user.id}"
        await achievements.register_member(chat_id, user.id, username, is_bot=user.is_bot)
