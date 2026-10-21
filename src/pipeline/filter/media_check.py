"""
MediaCheckNode — filter-stage node for non-text media.

Only explicitly addressed media reaches the filter: the router never sets
``should_respond`` for unaddressed media. A media message whose processing
produced no text gets an honest canned «не расслышал / не разглядел» reply —
generating a reaction from nothing is guaranteed hallucination. Transcribed
media is classified MEANINGFUL and then charged against the sender's attention
budget like a text turn, so switching to voice notes does not evade the
wind-down.
"""

import random

from src import log
from src.pipeline.state import BotState, ContentKind, Verdict

logger = log.get_logger(__name__)

# Honest canned acknowledgements for explicitly addressed media whose
# transcription/vision processing produced nothing — deterministic, so the
# response model never improvises a reaction to content it never perceived.
TRANSCRIPTION_FAILED_REPLIES = [
    "Не расслышал ни слова. Перезапиши или скинь текстом.",
    "Войс не прожевался. Повтори?",
    "У меня уши отвалились — что там было?",
    "Звук до меня не доехал. Ещё раз?",
]

VISION_FAILED_REPLIES = [
    "Не разглядел, что там. Перезалей?",
    "Картинка до меня не доехала. Скинь ещё раз или расскажи словами.",
]


class MediaCheckNode:
    """Pass transcribed media on to the budget; degrade honestly when not."""

    async def __call__(self, state: BotState) -> dict:
        """Check whether the media was transcribed or described.

        Args:
            state: Current pipeline state with a non-text incoming message.

        Returns:
            ``{"filter_verdict": "MEANINGFUL"}`` when there is text; otherwise
            a canned failure reply with ``should_respond=False``.
        """
        media_type = state["incoming"]["media_type"]
        text = state["incoming"]["processed_text"] or ""
        if text.strip():
            return {"filter_verdict": Verdict.MEANINGFUL}
        logger.info(
            "Filter[media_check]: no transcription for %s message %s — canned failure reply",
            media_type,
            state["incoming"]["message_id"],
        )
        pool = VISION_FAILED_REPLIES if media_type == "photo" else TRANSCRIPTION_FAILED_REPLIES
        return {
            "should_respond": False,
            "response": random.choice(pool),
            "content_kind": ContentKind.MEDIA_FAILURE_NOTICE,
        }
