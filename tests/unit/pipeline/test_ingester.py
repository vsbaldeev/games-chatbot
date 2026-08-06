"""MessageIngester tests — text-message path only (Shorts + social-link triggers).

Characterizes the existing Shorts behavior first (no test file covered this
before), then adds the new social-link behavior alongside it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipeline.ingester import MessageIngester, summarize_social_link
from tests.builders import make_incoming, make_state

SUMMARIZE_SHORT_TARGET = "src.pipeline.ingester.summarize_youtube_short"
SUMMARIZE_SOCIAL_LINK_TARGET = "src.pipeline.ingester.summarize_social_link"
UPDATE_CONTENT_TARGET = "src.pipeline.ingester.unified_messages.update_content"
HANDLERS_TARGET = "src.pipeline.ingester.social_links.HANDLERS"


@pytest.fixture
def ingester() -> MessageIngester:
    return MessageIngester()


class TestShortsIngestionCharacterization:
    """Pins today's behavior before Step 3 refactors __ingest_text/__call__."""

    async def test_successful_short_appends_block_and_sets_content(self, ingester):
        incoming = make_incoming(raw_text="check this out https://www.youtube.com/shorts/abc123")
        state = make_state(
            incoming, should_respond=True, response_trigger="youtube_short",
            youtube_short_url="https://www.youtube.com/shorts/abc123",
        )
        with (
            patch(SUMMARIZE_SHORT_TARGET, new=AsyncMock(return_value="[YouTube Shorts]\nsome content")),
            patch(UPDATE_CONTENT_TARGET, new=AsyncMock()),
        ):
            result = await ingester(state)
        assert "[YouTube Shorts]\nsome content" in result["incoming"]["processed_text"]
        assert result["youtube_short_content"] == "[YouTube Shorts]\nsome content"

    async def test_failed_short_leaves_raw_text_unchanged(self, ingester):
        incoming = make_incoming(raw_text="check this out https://www.youtube.com/shorts/abc123")
        state = make_state(
            incoming, should_respond=True, response_trigger="youtube_short",
            youtube_short_url="https://www.youtube.com/shorts/abc123",
        )
        with patch(SUMMARIZE_SHORT_TARGET, new=AsyncMock(return_value="")):
            result = await ingester(state)
        assert result["incoming"]["processed_text"] == incoming["raw_text"]
        # Post-refactor contract: extra fields are merged only on success, so
        # a failed fetch omits the key entirely (state.py's NotRequired and
        # filter_node.py's state.get(...) both already treat absent and
        # None-valued identically) — the pre-refactor code always set the
        # key to None on this trigger, which this pinning test originally
        # asserted; that assertion is updated here to the refactor's actual
        # (and intentional) shape.
        assert result.get("youtube_short_content") is None


class TestSocialLinkIngestion:
    async def test_successful_fetch_appends_block_and_sets_content_and_video(self, ingester):
        incoming = make_incoming(raw_text="look https://www.instagram.com/reel/abc123/")
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_handler="instagram_reel",
            social_link_url="https://www.instagram.com/reel/abc123/",
        )
        with (
            patch(
                SUMMARIZE_SOCIAL_LINK_TARGET,
                new=AsyncMock(return_value=("[Instagram Reel]\ncaption here", b"video bytes")),
            ),
            patch(UPDATE_CONTENT_TARGET, new=AsyncMock()),
        ):
            result = await ingester(state)
        assert "[Instagram Reel]\ncaption here" in result["incoming"]["processed_text"]
        assert result["social_link_content"] == "[Instagram Reel]\ncaption here"
        assert result["social_link_video"] == b"video bytes"

    async def test_reddit_fetch_has_no_video(self, ingester):
        incoming = make_incoming(raw_text="look https://www.reddit.com/r/x/comments/abc/y/")
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_handler="reddit_post",
            social_link_url="https://www.reddit.com/r/x/comments/abc/y/",
        )
        with (
            patch(
                SUMMARIZE_SOCIAL_LINK_TARGET,
                new=AsyncMock(return_value=("[Reddit r/x] «title»", None)),
            ),
            patch(UPDATE_CONTENT_TARGET, new=AsyncMock()),
        ):
            result = await ingester(state)
        assert result["social_link_content"] == "[Reddit r/x] «title»"
        assert result["social_link_video"] is None

    async def test_failed_fetch_leaves_raw_text_unchanged_and_omits_keys(self, ingester):
        incoming = make_incoming(raw_text="look https://www.reddit.com/r/x/comments/abc/y/")
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_handler="reddit_post",
            social_link_url="https://www.reddit.com/r/x/comments/abc/y/",
        )
        with patch(SUMMARIZE_SOCIAL_LINK_TARGET, new=AsyncMock(return_value=("", None))):
            result = await ingester(state)
        assert result["incoming"]["processed_text"] == incoming["raw_text"]
        assert result.get("social_link_content") is None
        assert result.get("social_link_video") is None


class TestSummarizeSocialLink:
    """Exercises summarize_social_link itself (not mocked out), per handler."""

    async def test_unknown_handler_name_returns_empty(self):
        content_block, video_bytes = await summarize_social_link("nonexistent_handler", "url")
        assert (content_block, video_bytes) == ("", None)

    async def test_handler_fetch_raising_degrades_to_empty(self):
        broken_handler = MagicMock()
        broken_handler.name = "reddit_post"
        broken_handler.fetch = AsyncMock(side_effect=ValueError("malformed payload shape"))
        with patch(HANDLERS_TARGET, [broken_handler]):
            content_block, video_bytes = await summarize_social_link(
                "reddit_post", "https://www.reddit.com/comments/abc"
            )
        assert (content_block, video_bytes) == ("", None)

    async def test_handler_fetch_success_returns_content_and_video(self):
        working_handler = MagicMock()
        working_handler.name = "instagram_reel"
        working_handler.fetch = AsyncMock(
            return_value={"content_block": "[Instagram Reel]\ncaption", "video_bytes": b"bytes"}
        )
        with patch(HANDLERS_TARGET, [working_handler]):
            content_block, video_bytes = await summarize_social_link(
                "instagram_reel", "https://www.instagram.com/reel/abc123/"
            )
        assert content_block == "[Instagram Reel]\ncaption"
        assert video_bytes == b"bytes"
