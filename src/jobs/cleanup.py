"""Daily cleanup job — prunes old rows from unified_messages and thread_history."""

from src import log
from src.store import thread_history, unified_messages

logger = log.get_logger(__name__)


async def cleanup_messages_job(context) -> None:
    deleted_messages = await unified_messages.cleanup_old()
    logger.info(
        "Cleanup: deleted %d old messages (retention=%dd)",
        deleted_messages, unified_messages.MESSAGE_RETENTION_DAYS,
    )
    deleted_history = await thread_history.cleanup_old()
    logger.info(
        "Cleanup: deleted %d old thread history rows (retention=%dd)",
        deleted_history, thread_history.HISTORY_RETENTION_DAYS,
    )
