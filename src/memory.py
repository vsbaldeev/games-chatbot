import asyncio

from langchain_community.chat_message_histories import SQLChatMessageHistory

from src import config

DB_URL = f"sqlite:///{config.SQLITE_DB_PATH}"


def get_chat_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=DB_URL,
        table_name="message_store",
    )


async def trim_history(history: SQLChatMessageHistory, max_messages: int) -> None:
    """Trim history to the last max_messages entries. Runs sync SQLAlchemy calls in a thread."""
    def trim_sync() -> None:
        messages = history.messages
        if len(messages) <= max_messages:
            return
        messages_to_keep = messages[-max_messages:]
        history.clear()
        for msg in messages_to_keep:
            history.add_message(msg)

    await asyncio.to_thread(trim_sync)


async def get_recent_messages(history: SQLChatMessageHistory, count: int) -> list:
    """Fetch the last `count` messages without blocking the event loop."""
    return await asyncio.to_thread(lambda: history.messages[-count:])
