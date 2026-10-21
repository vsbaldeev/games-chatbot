"""check_link tests: Shorts and social-link dispatch."""

from unittest.mock import MagicMock

from src.pipeline.filter.link_check import (
    SHORTS_FAILED_REPLIES,
    SOCIAL_LINK_FAILED_REPLIES,
    check_link,
)
from src.pipeline.state import ContentKind
from tests.builders import make_incoming, make_state


class TestYoutubeShortDispatchCharacterization:
    """Shorts-trigger dispatch."""

    async def test_content_present_sets_shorts_verdict(self):
        incoming = make_incoming(raw_text="https://www.youtube.com/shorts/abc")
        state = make_state(
            incoming, should_respond=True, response_trigger="youtube_short",
            youtube_short_content="[YouTube Shorts]\nsome content",
        )
        result = await check_link(state)
        assert result == {"filter_verdict": "SHORTS"}

    async def test_no_content_and_unaddressed_drops_silently(self):
        telegram_message = MagicMock(text="https://www.youtube.com/shorts/abc", caption=None)
        telegram_message.reply_to_message = None
        incoming = make_incoming(
            raw_text="https://www.youtube.com/shorts/abc", telegram_message=telegram_message,
        )
        state = make_state(
            incoming, should_respond=True, response_trigger="youtube_short",
            youtube_short_content=None,
        )
        result = await check_link(state)
        assert result == {"should_respond": False, "drop_reason": "shorts_failed"}

    async def test_no_content_and_addressed_gets_canned_media_failure_notice(self):
        telegram_message = MagicMock(text="@testbot https://www.youtube.com/shorts/abc", caption=None)
        telegram_message.reply_to_message = None
        incoming = make_incoming(
            raw_text="@testbot https://www.youtube.com/shorts/abc", telegram_message=telegram_message,
        )
        state = make_state(
            incoming, should_respond=True, response_trigger="youtube_short",
            youtube_short_content=None,
        )
        result = await check_link(state)
        assert result["should_respond"] is False
        assert result["response"] in SHORTS_FAILED_REPLIES
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE


class TestSocialLinkDispatch:
    async def test_content_present_sets_social_link_verdict(self):
        incoming = make_incoming(raw_text="https://www.reddit.com/r/x/comments/abc/y/")
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_content="[Reddit r/x] «title»",
        )
        result = await check_link(state)
        assert result == {"filter_verdict": "SOCIAL_LINK"}

    async def test_no_content_and_unaddressed_drops_silently(self):
        telegram_message = MagicMock(text="https://www.reddit.com/r/x/comments/abc/y/", caption=None)
        telegram_message.reply_to_message = None
        incoming = make_incoming(
            raw_text="https://www.reddit.com/r/x/comments/abc/y/", telegram_message=telegram_message,
        )
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_content=None,
        )
        result = await check_link(state)
        assert result == {"should_respond": False, "drop_reason": "social_link_failed"}

    async def test_no_content_and_addressed_gets_canned_reply(self):
        telegram_message = MagicMock(
            text="@testbot https://www.reddit.com/r/x/comments/abc/y/", caption=None,
        )
        telegram_message.reply_to_message = None
        incoming = make_incoming(
            raw_text="@testbot https://www.reddit.com/r/x/comments/abc/y/",
            telegram_message=telegram_message,
        )
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_content=None,
        )
        result = await check_link(state)
        assert result["should_respond"] is False
        assert result["response"] in SOCIAL_LINK_FAILED_REPLIES
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE
