"""
MessageRouter — first node in the LangGraph pipeline.

Responsibilities:
  1. Persist every incoming message to unified_messages (as text or placeholder).
  2. Decide whether the bot should respond (sets should_respond).

Respond when:
  - The bot is @mentioned in the text / caption.
  - The message is a direct reply to a bot message.
  - Random chance fires for voice (25%), video_note (25%), or photo (25%).
"""

import random
from src import log
from typing import Any

from src.pipeline.state import BotState, IncomingMessage
from src.store import unified_messages

logger = log.get_logger(__name__)

VOICE_RESPONSE_CHANCE = 0.25
PHOTO_RESPONSE_CHANCE = 0.25


class MessageRouter:
    """Determines whether the bot should respond and writes the message to the store."""

    def __init__(self, bot_username: str) -> None:
        self.__bot_username = bot_username.lower()

    async def __call__(self, state: BotState) -> dict:
        msg: IncomingMessage = state["incoming"]
        update = msg["update"]
        message = update.message

        await self.__store_message(msg)

        should_respond = self.__decide(msg, message)
        return {"should_respond": should_respond}

    async def __store_message(self, msg: IncomingMessage) -> None:
        media_type = msg["media_type"]

        if media_type == "text":
            content = msg["raw_text"] or ""
        elif media_type == "voice":
            content = unified_messages.VOICE_PLACEHOLDER
        elif media_type == "video_note":
            content = unified_messages.VIDEO_NOTE_PLACEHOLDER
        elif media_type == "video":
            content = unified_messages.VIDEO_PLACEHOLDER
        elif media_type == "photo":
            content = unified_messages.PHOTO_PLACEHOLDER
        else:
            content = ""

        try:
            await unified_messages.insert(
                chat_id=msg["chat_id"],
                message_id=msg["message_id"],
                user_id=msg["user_id"],
                username=msg["username"],
                content=content,
                media_type=media_type,
                reply_to_msg_id=msg["reply_to_msg_id"],
                file_id=msg["file_id"],
            )
        except Exception as err:
            logger.warning("Failed to store message %s: %s", msg["message_id"], err)

    def __decide(self, msg: IncomingMessage, telegram_message: Any) -> bool:
        media_type = msg["media_type"]

        if media_type == "text":
            text = msg["raw_text"] or ""
            if self.__bot_username in text.lower():
                return True
            if self.__is_reply_to_bot(telegram_message):
                return True
            return False

        if media_type in ("voice", "video_note", "video"):
            return random.random() < VOICE_RESPONSE_CHANCE

        if media_type == "photo":
            caption = (telegram_message.caption or "").lower()
            if self.__bot_username in caption:
                return True
            return random.random() < PHOTO_RESPONSE_CHANCE

        return False

    @staticmethod
    def __is_reply_to_bot(telegram_message: Any) -> bool:
        reply = getattr(telegram_message, "reply_to_message", None)
        if not reply:
            return False
        sender = getattr(reply, "from_user", None)
        return sender is not None and getattr(sender, "is_bot", False)
