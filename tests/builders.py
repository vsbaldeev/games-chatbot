"""Helpers for constructing IncomingMessage and BotState dicts in tests."""

from unittest.mock import MagicMock


def make_telegram_message(
    *,
    reply_to_user_id: int | None = None,
    caption: str | None = None,
) -> MagicMock:
    telegram_message = MagicMock()
    telegram_message.caption = caption
    if reply_to_user_id is not None:
        reply = MagicMock()
        reply.from_user = MagicMock()
        reply.from_user.id = reply_to_user_id
        telegram_message.reply_to_message = reply
    else:
        telegram_message.reply_to_message = None
    return telegram_message


def make_incoming(
    *,
    chat_id: int = 1000,
    user_id: int = 42,
    username: str = "testuser",
    raw_text: str | None = "hello world",
    processed_text: str | None = None,
    media_type: str = "text",
    message_id: int = 100,
    reply_to_msg_id: int | None = None,
    file_id: str | None = None,
    is_forwarded: bool = False,
    media_group_id: str | None = None,
    telegram_message: MagicMock | None = None,
) -> dict:
    if telegram_message is None:
        telegram_message = make_telegram_message()
    update = MagicMock()
    update.message = telegram_message
    return {
        "update": update,
        "chat_id": chat_id,
        "user_id": user_id,
        "username": username,
        "raw_text": raw_text,
        "processed_text": processed_text,
        "media_type": media_type,
        "message_id": message_id,
        "reply_to_msg_id": reply_to_msg_id,
        "file_id": file_id,
        "is_forwarded": is_forwarded,
        "media_group_id": media_group_id,
    }


def make_state(incoming: dict, **overrides) -> dict:
    state = {
        "incoming": incoming,
        "should_respond": False,
        "response_trigger": "random",
        "blocked": False,
        "context": None,
        "response": None,
        "context_types": MagicMock(),
    }
    state.update(overrides)
    return state


def make_message_row(
    *,
    message_id: int,
    username: str = "alice",
    user_id: int = 1,
    content: str = "hi",
    media_type: str = "text",
) -> dict:
    return {
        "message_id": message_id,
        "username": username,
        "user_id": user_id,
        "content": content,
        "media_type": media_type,
    }
