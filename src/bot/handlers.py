"""Handler managers — each class registers a group of handlers on the Telegram Application."""

from abc import ABC, abstractmethod

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    MessageHandler,
    MessageReactionHandler,
    TypeHandler,
    filters,
)

from src.commands import general, fun, games, statistics
from src.events.members import (
    track_member,
    handle_new_chat_members,
    handle_bot_added_to_chat,
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
from src.events.reactions import handle_reaction


class HandlerManagerInterface(ABC):
    @abstractmethod
    def add_handlers(self, app: Application) -> None: ...


class EventHandlerManager(HandlerManagerInterface):
    def add_handlers(self, app: Application) -> None:
        app.add_handler(TypeHandler(Update, track_member), group=-1)
        app.add_handler(MessageHandler(
            filters.StatusUpdate.NEW_CHAT_MEMBERS & filters.ChatType.GROUPS,
            handle_new_chat_members,
        ))
        app.add_handler(ChatMemberHandler(
            handle_bot_added_to_chat, ChatMemberHandler.MY_CHAT_MEMBER,
        ))
        app.add_handler(MessageReactionHandler(
            handle_reaction,
            message_reaction_types=MessageReactionHandler.MESSAGE_REACTION_UPDATED,
        ))


class CommandHandlerManager(HandlerManagerInterface):
    def add_handlers(self, app: Application) -> None:
        group_only = filters.ChatType.GROUPS
        app.add_handler(CommandHandler("start", general.cmd_start, filters=group_only))
        app.add_handler(CommandHandler("help", general.cmd_help, filters=group_only))
        app.add_handler(CommandHandler("roast", fun.cmd_roast, filters=group_only))
        app.add_handler(CommandHandler("meme", fun.cmd_meme, filters=group_only))
        app.add_handler(CommandHandler("duel", games.cmd_duel, filters=group_only))
        app.add_handler(CallbackQueryHandler(games.handle_duel_callback, pattern=games.DUEL_CALLBACK_PATTERN))
        app.add_handler(CommandHandler("dnd_pvp", games.cmd_dnd_pvp, filters=group_only))
        app.add_handler(CommandHandler("dnd_coop", games.cmd_dnd_coop, filters=group_only))
        app.add_handler(CommandHandler("dnd_heist", games.cmd_dnd_heist, filters=group_only))
        app.add_handler(CallbackQueryHandler(games.handle_dnd_callback, pattern=games.DND_CALLBACK_PATTERN))
        app.add_handler(CommandHandler("achievements", statistics.cmd_achievements, filters=group_only))
        app.add_handler(CommandHandler("top", statistics.cmd_top, filters=group_only))


class MessageHandlerManager(HandlerManagerInterface):
    def add_handlers(self, app: Application) -> None:
        group_only = filters.ChatType.GROUPS
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND & group_only,
            handle_message,
        ))
        app.add_handler(MessageHandler(
            (filters.VOICE | filters.VIDEO_NOTE) & group_only,
            handle_voice_message,
        ))
        app.add_handler(MessageHandler(
            filters.PHOTO & group_only,
            handle_photo_message,
        ))
        app.add_handler(MessageHandler(
            filters.Sticker.ALL & group_only,
            handle_sticker_message,
        ))
        app.add_handler(MessageHandler(
            filters.VIDEO & group_only,
            handle_video_message,
        ))
        app.add_handler(MessageHandler(
            filters.ANIMATION & group_only,
            handle_animation_message,
        ))
        app.add_handler(MessageHandler(
            filters.AUDIO & group_only,
            handle_audio_message,
        ))
