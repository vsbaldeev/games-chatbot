"""check_link tests: Shorts and social-link dispatch."""

from unittest.mock import MagicMock

import pytest

from src.pipeline.ingest.filter.link_check import (
    SHORTS_FAILED_REPLIES,
    SOCIAL_LINK_FAILED_REPLIES,
    check_link,
)
from src.pipeline.state import ContentKind, Verdict
from tests.builders import make_incoming, make_state

LINK_TRIGGERS = pytest.mark.parametrize(
    "trigger, content_key, verdict, replies, drop_reason",
    [
        ("youtube_short", "youtube_short_content", Verdict.SHORTS,
         SHORTS_FAILED_REPLIES, "shorts_failed"),
        ("social_link", "social_link_content", Verdict.SOCIAL_LINK,
         SOCIAL_LINK_FAILED_REPLIES, "social_link_failed"),
    ],
    ids=["shorts", "social-link"],
)


def make_link_state(trigger: str, content_key: str, content: str | None, text: str) -> dict:
    telegram_message = MagicMock(text=text, caption=None)
    telegram_message.reply_to_message = None
    incoming = make_incoming(raw_text=text, telegram_message=telegram_message)
    return make_state(
        incoming, should_respond=True, response_trigger=trigger, **{content_key: content},
    )


@LINK_TRIGGERS
async def test_check_link_content_present_sets_the_trigger_verdict(
    trigger, content_key, verdict, replies, drop_reason
):
    state = make_link_state(trigger, content_key, "[summary]\nsome content", "https://example.com/x")
    assert await check_link(state) == {"filter_verdict": verdict}


@LINK_TRIGGERS
async def test_check_link_no_content_and_unaddressed_drops_silently(
    trigger, content_key, verdict, replies, drop_reason
):
    state = make_link_state(trigger, content_key, None, "https://example.com/x")
    assert await check_link(state) == {"should_respond": False, "drop_reason": drop_reason}


@LINK_TRIGGERS
async def test_check_link_no_content_and_addressed_gets_canned_media_failure_notice(
    trigger, content_key, verdict, replies, drop_reason
):
    state = make_link_state(trigger, content_key, None, "@testbot https://example.com/x")
    result = await check_link(state)
    assert result["should_respond"] is False
    assert result["response"] in replies
    assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE
