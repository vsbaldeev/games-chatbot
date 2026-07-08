"""Daily cleanup job — prunes old messages, thread history and stale memory facts."""

from src import log
from src.store import thread_history, unified_messages, user_memories

logger = log.get_logger(__name__)


async def cleanup_messages_job(context) -> None:
    """Run the nightly retention sweep across all aging stores.

    Args:
        context: Telegram JobQueue callback context (unused).
    """
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
    deleted_facts = await user_memories.cleanup_stale()
    logger.info(
        "Cleanup: deleted %d stale memory facts (retention=%dd)",
        deleted_facts, user_memories.FACT_RETENTION_DAYS,
    )
