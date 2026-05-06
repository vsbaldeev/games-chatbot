"""Daily cleanup job — prunes old rows from unified_messages."""

from src import log
from src.store import unified_messages

logger = log.get_logger(__name__)


async def cleanup_messages_job(context) -> None:
    deleted = await unified_messages.cleanup_old()
    logger.info("Cleanup: deleted %d old messages (retention=%dd)", deleted, unified_messages.MESSAGE_RETENTION_DAYS)
