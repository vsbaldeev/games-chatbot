"""Scheduled job: close 15-minute feedback windows, then roll closed windows
up into hourly bot_feedback_metrics rows.

Runs every 5 minutes (see FeedbackJobManager, src/app/jobs.py) — frequent
enough that a message's ignored/corrected verdict and its metrics bucket
both settle within a few minutes of the window actually closing.
"""

from src import log
from src.store import feedback_metrics, message_feedback

logger = log.get_logger(__name__)


async def feedback_metrics_job(context) -> None:
    """Close expired feedback windows and recompute recent hourly buckets.

    Args:
        context: Telegram JobQueue callback context (unused).
    """
    try:
        closed = await message_feedback.close_expired_windows()
        logger.info("Feedback: closed %d expired feedback windows", closed)
    except Exception as err:
        logger.warning("Failed to close expired feedback windows: %s", err)
        return
    try:
        written = await feedback_metrics.recompute_recent_buckets()
        logger.info("Feedback: wrote %d metrics buckets", written)
    except Exception as err:
        logger.warning("Failed to recompute feedback metrics buckets: %s", err)
