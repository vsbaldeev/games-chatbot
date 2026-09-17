"""Daily cleanup job — prunes old messages, thread history, stale memory
facts, and the feedback-metrics tables."""

from src import log
from src.store import feedback_metrics, llm_log, llm_system_prompts, message_feedback, thread_history, unified_messages, user_memories

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
    deleted_llm_log = await llm_log.cleanup_old()
    logger.info(
        "Cleanup: deleted %d old LLM log rows (retention=%dd)",
        deleted_llm_log, llm_log.RETENTION_DAYS,
    )
    # Must run after llm_log's own cleanup — a prompt is only unreferenced
    # once every bot_llm_log row citing it is already gone.
    deleted_prompts = await llm_system_prompts.cleanup_unreferenced()
    logger.info("Cleanup: deleted %d unreferenced system prompts", deleted_prompts)
    deleted_feedback = await message_feedback.cleanup_old()
    logger.info(
        "Cleanup: deleted %d old feedback rows (retention=%dd)",
        deleted_feedback, message_feedback.RETENTION_DAYS,
    )
    deleted_metrics = await feedback_metrics.cleanup_old()
    logger.info(
        "Cleanup: deleted %d old feedback metrics rows (retention=%dd)",
        deleted_metrics, feedback_metrics.RETENTION_DAYS,
    )
