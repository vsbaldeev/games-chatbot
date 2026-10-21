"""
LinkCheckNode — filter-stage node for Shorts and social-link triggers.

YouTube Shorts (``response_trigger="youtube_short"``) and Instagram/YouTube
links (``response_trigger="social_link"``) bypass LLM classification entirely:
a bare link would be classified MEANINGLESS and dropped, but the trigger is
deterministic. A successful fetch passes through; a failed one gets a canned
«не смог посмотреть» reply when the sender explicitly addressed the bot, and
full silence otherwise (no emoji reaction — the bot was never addressed).
"""

import random

from src import config, log
from src.pipeline.router import is_explicitly_addressed
from src.pipeline.state import BotState, ContentKind, Verdict

logger = log.get_logger(__name__)

# Honest canned acknowledgements for an explicitly addressed Shorts link the
# bot failed to download or extract content from — same principle as the
# transcription-failure pools: never improvise a reaction to unseen content.
SHORTS_FAILED_REPLIES = [
    "Не смог посмотреть этот шортс — ютуб зажал. Смотрите сами.",
    "Ролик не открылся. Придётся рискнуть и смотреть вслепую.",
    "Ютуб мне этот шортс не отдал. Что я, впервые ему не угодил.",
    "Не дотянулся до ролика. Перекиньте другую ссылку или смотрите так.",
]

# Honest canned acknowledgements for an explicitly addressed Instagram/YouTube
# link the bot failed to fetch — same principle as SHORTS_FAILED_REPLIES.
SOCIAL_LINK_FAILED_REPLIES = [
    "Не смог посмотреть — сайт не отдал. Сами гляньте по ссылке.",
    "Ссылка не открылась. Придётся смотреть вслепую.",
    "Не дотянулся до контента. Перекиньте ещё раз или смотрите так.",
]


class LinkCheckNode:
    """Pass a fetched link summary through; degrade honestly when the fetch failed."""

    async def __call__(self, state: BotState) -> dict:
        """Dispatch on the link trigger.

        Args:
            state: Current pipeline state; ``response_trigger`` is
                ``"youtube_short"`` or ``"social_link"``.

        Returns:
            State update dict.
        """
        if state.get("response_trigger") == "youtube_short":
            return self.__handle_youtube_short(state)
        return self.__handle_social_link(state)

    def __handle_youtube_short(self, state: BotState) -> dict:
        """Pass a summarized Shorts link through; degrade honestly on failure.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict.
        """
        if state.get("youtube_short_content"):
            return {"filter_verdict": Verdict.SHORTS}
        message_id = state["incoming"]["message_id"]
        if self.__is_addressed(state):
            logger.info(
                "Filter[link_check]: no Shorts content for message %s, explicit trigger — "
                "canned failure reply",
                message_id,
            )
            return self.__canned_failure(SHORTS_FAILED_REPLIES)
        logger.info(
            "Filter[link_check]: no Shorts content for message %s, skipping silently", message_id
        )
        return {"should_respond": False, "drop_reason": "shorts_failed"}

    def __handle_social_link(self, state: BotState) -> dict:
        """Pass a fetched Instagram/YouTube summary through; degrade honestly on failure.

        Kept as its own method rather than merged with ``__handle_youtube_short``:
        the two triggers are handled by entirely separate code paths by design
        (see the design doc's "Why Shorts stays separate" section).

        Args:
            state: Current pipeline state.

        Returns:
            State update dict.
        """
        if state.get("social_link_content"):
            return {"filter_verdict": Verdict.SOCIAL_LINK}
        message_id = state["incoming"]["message_id"]
        if self.__is_addressed(state):
            logger.info(
                "Filter[link_check]: no social-link content for message %s, explicit trigger — "
                "canned failure reply",
                message_id,
            )
            return self.__canned_failure(SOCIAL_LINK_FAILED_REPLIES)
        logger.info(
            "Filter[link_check]: no social-link content for message %s, skipping silently",
            message_id,
        )
        return {"should_respond": False, "drop_reason": "social_link_failed"}

    @staticmethod
    def __is_addressed(state: BotState) -> bool:
        telegram_message = state["incoming"]["update"].message
        return telegram_message is not None and is_explicitly_addressed(
            telegram_message, config.BOT_USERNAME, config.BOT_ID
        )

    @staticmethod
    def __canned_failure(pool: list[str]) -> dict:
        return {
            "should_respond": False,
            "response": random.choice(pool),
            "content_kind": ContentKind.MEDIA_FAILURE_NOTICE,
        }
