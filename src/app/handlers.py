"""Handler managers — each class registers a group of handlers on the Telegram Application."""

from abc import ABC, abstractmethod

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    TypeHandler,
    filters,
)

from src.commands import general, games
from src.events.members import (
    register_sender_as_member,
    register_users_from_join_message,
)
from src.events.messages import (
    handle_message,
    handle_voice_message,
    handle_photo_message,
    handle_sticker_message,
    handle_video_message,
    handle_animation_message,
    handle_audio_message,
)

class HandlerManagerInterface(ABC):
    @abstractmethod
    def add_handlers(self, app: Application) -> None: ...


class EventHandlerManager(HandlerManagerInterface):
    def add_handlers(self, app: Application) -> None:
        app.add_handler(TypeHandler(Update, register_sender_as_member), group=-1)
        app.add_handler(MessageHandler(
            filters.StatusUpdate.NEW_CHAT_MEMBERS & filters.ChatType.GROUPS,
            register_users_from_join_message,
        ))


class CommandHandlerManager(HandlerManagerInterface):
    def add_handlers(self, app: Application) -> None:
        group_only = filters.ChatType.GROUPS
        app.add_handler(CommandHandler("start", general.cmd_start, filters=group_only))
        app.add_handler(CommandHandler("help", general.cmd_help, filters=group_only))
        app.add_handler(CommandHandler("duel", games.cmd_duel, filters=group_only))
        app.add_handler(CallbackQueryHandler(games.handle_duel_callback, pattern=games.DUEL_CALLBACK_PATTERN))


class MessageHandlerManager(HandlerManagerInterface):
    def add_handlers(self, app: Application) -> None:
        group_only = filters.ChatType.GROUPS
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & group_only, handle_message))
        app.add_handler(MessageHandler((filters.VOICE | filters.VIDEO_NOTE) & group_only, handle_voice_message))
        app.add_handler(MessageHandler(filters.PHOTO & group_only, handle_photo_message))
        app.add_handler(MessageHandler(filters.Sticker.ALL & group_only, handle_sticker_message))
        app.add_handler(MessageHandler(filters.VIDEO & group_only, handle_video_message))
        app.add_handler(MessageHandler(filters.ANIMATION & group_only, handle_animation_message))
        app.add_handler(MessageHandler(filters.AUDIO & group_only, handle_audio_message))
