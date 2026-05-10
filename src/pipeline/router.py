"""
MessageRouter — first node in the LangGraph pipeline.

Responsibilities:
  1. Persist every incoming message to unified_messages (as text or placeholder).
  2. Decide whether the bot should respond (sets should_respond).
  3. Fire passive memory extraction for long plain-text messages that won't get a response.

Respond when:
  - The bot is @mentioned in the text / caption.
  - The message is a direct reply to a bot message (text or any media type).
  - Random chance fires for voice/video_note/video/photo (25%).
"""

import asyncio
import random
from typing import Any

from src import log
from src.pipeline.memory_writer import MIN_PASSIVE_LENGTH, extract_and_save
from src.pipeline.state import BotState, IncomingMessage
from src.store import unified_messages

logger = log.get_logger(__name__)

MEDIA_RESPONSE_CHANCE = 0.25


class MessageRouter:
    """Determines whether the bot should respond and writes the message to the store."""

    def __init__(self, bot_username: str, bot_id: int) -> None:
        self.__bot_username = bot_username.lower()
        self.__bot_id = bot_id

    async def __call__(self, state: BotState) -> dict:
        msg: IncomingMessage = state["incoming"]
        update = msg["update"]
        message = update.message

        await self.__store_message(msg)

        should_respond, response_trigger = self.__decide(msg, message)

        if not should_respond and not msg.get("is_forwarded") and msg["media_type"] == "text":
            text = msg["raw_text"] or ""
            if len(text.strip()) >= MIN_PASSIVE_LENGTH:
                asyncio.create_task(extract_and_save(
                    chat_id=msg["chat_id"],
                    user_id=msg["user_id"],
                    username=msg["username"],
                    user_message=text,
                ))

        return {"should_respond": should_respond, "response_trigger": response_trigger}

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
            content = msg["raw_text"] or unified_messages.PHOTO_PLACEHOLDER
        elif media_type == "sticker":
            content = unified_messages.STICKER_PLACEHOLDER
        elif media_type == "animation":
            content = unified_messages.ANIMATION_PLACEHOLDER
        elif media_type == "audio":
            content = unified_messages.AUDIO_PLACEHOLDER
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

    def __decide(self, msg: IncomingMessage, telegram_message: Any) -> tuple[bool, str]:
        media_type = msg["media_type"]

        if media_type == "text":
            text = msg["raw_text"] or ""
            if self.__bot_username in text.lower():
                return True, "explicit"
            if self.__is_reply_to_bot(telegram_message):
                return True, "explicit"
            return False, "random"

        if media_type in ("voice", "video_note", "video", "photo"):
            caption = (getattr(telegram_message, "caption", None) or "").lower()
            if self.__bot_username in caption or self.__is_reply_to_bot(telegram_message):
                return True, "explicit"
            return random.random() < MEDIA_RESPONSE_CHANCE, "random"

        return False, "random"

    def __is_reply_to_bot(self, telegram_message: Any) -> bool:
        reply = getattr(telegram_message, "reply_to_message", None)
        if not reply:
            return False
        sender = getattr(reply, "from_user", None)
        return sender is not None and sender.id == self.__bot_id
