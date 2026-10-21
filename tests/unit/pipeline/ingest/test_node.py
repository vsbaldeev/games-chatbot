"""MessageIngester tests: text path (Shorts + social links) and the media-type strategies."""

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline.ingest.node import MessageIngester
from tests.builders import make_incoming, make_state

SUMMARIZE_SHORT_TARGET = "src.pipeline.ingest.node.summarize_youtube_short"
SUMMARIZE_SOCIAL_LINK_TARGET = "src.pipeline.ingest.node.summarize_social_link"
UPDATE_CONTENT_TARGET = "src.pipeline.ingest.node.unified_messages.update_content"
DESCRIBE_STICKER_TARGET = "src.pipeline.ingest.node.describe_sticker"


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
            patch(
                SUMMARIZE_SHORT_TARGET,
                new=AsyncMock(return_value=("[YouTube Shorts]\nsome content", b"video bytes")),
            ),
            patch(UPDATE_CONTENT_TARGET, new=AsyncMock()),
        ):
            result = await ingester(state)
        assert "[YouTube Shorts]\nsome content" in result["incoming"]["processed_text"]
        assert result["youtube_short_content"] == "[YouTube Shorts]\nsome content"
        assert result["youtube_short_video"] == b"video bytes"

    async def test_failed_short_leaves_raw_text_unchanged(self, ingester):
        incoming = make_incoming(raw_text="check this out https://www.youtube.com/shorts/abc123")
        state = make_state(
            incoming, should_respond=True, response_trigger="youtube_short",
            youtube_short_url="https://www.youtube.com/shorts/abc123",
        )
        with patch(SUMMARIZE_SHORT_TARGET, new=AsyncMock(return_value=("", None))):
            result = await ingester(state)
        assert result["incoming"]["processed_text"] == incoming["raw_text"]
        assert result.get("youtube_short_content") is None
        assert result.get("youtube_short_video") is None


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

    async def test_youtube_fetch_has_no_video(self, ingester):
        incoming = make_incoming(raw_text="look https://www.youtube.com/watch?v=abc123")
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_handler="youtube_video",
            social_link_url="https://www.youtube.com/watch?v=abc123",
        )
        with (
            patch(
                SUMMARIZE_SOCIAL_LINK_TARGET,
                new=AsyncMock(return_value=("[YouTube] «title»", None)),
            ),
            patch(UPDATE_CONTENT_TARGET, new=AsyncMock()),
        ):
            result = await ingester(state)
        assert result["social_link_content"] == "[YouTube] «title»"
        assert result["social_link_video"] is None

    async def test_failed_fetch_leaves_raw_text_unchanged_and_omits_keys(self, ingester):
        incoming = make_incoming(raw_text="look https://www.youtube.com/watch?v=abc123")
        state = make_state(
            incoming, should_respond=True, response_trigger="social_link",
            social_link_handler="youtube_video",
            social_link_url="https://www.youtube.com/watch?v=abc123",
        )
        with patch(SUMMARIZE_SOCIAL_LINK_TARGET, new=AsyncMock(return_value=("", None))):
            result = await ingester(state)
        assert result["incoming"]["processed_text"] == incoming["raw_text"]
        assert result.get("social_link_content") is None
        assert result.get("social_link_video") is None


class TestLinkMaterialIsNotPersistedAsChatContent:
    """Fetched link material must stay out of the stored chat row.

    Observed in production: two Reels posted back to back made the bot
    describe both. The first Reel's row had been rewritten to
    ``link + [Instagram Reel] block``, so the second Reel's prompt carried
    two indistinguishable material blocks in its recent-history window. The
    material belongs to this run's prompt and to the ``link_material``
    column, never to the message's chat content.
    """

    @pytest.mark.parametrize(
        "trigger_fields, summarize_target, content_block",
        [
            (
                {
                    "response_trigger": "social_link",
                    "social_link_handler": "instagram_reel",
                    "social_link_url": "https://www.instagram.com/reel/abc123/",
                },
                SUMMARIZE_SOCIAL_LINK_TARGET,
                "[Instagram Reel]\ncaption here",
            ),
            (
                {
                    "response_trigger": "youtube_short",
                    "youtube_short_url": "https://www.youtube.com/shorts/abc123",
                },
                SUMMARIZE_SHORT_TARGET,
                "[YouTube Shorts]\nsome content",
            ),
        ],
        ids=["social-link", "youtube-short"],
    )
    async def test_successful_fetch_leaves_stored_row_untouched(
        self, ingester, trigger_fields, summarize_target, content_block
    ):
        incoming = make_incoming(raw_text="look https://example.com/link")
        state = make_state(incoming, should_respond=True, **trigger_fields)
        with (
            patch(summarize_target, new=AsyncMock(return_value=(content_block, b"video bytes"))),
            patch(UPDATE_CONTENT_TARGET, new=AsyncMock()) as update_content,
        ):
            result = await ingester(state)
        assert content_block in result["incoming"]["processed_text"]
        update_content.assert_not_awaited()


class TestMediaTypeStrategies:
    async def test_unaddressed_sticker_makes_no_vision_call(self, ingester):
        incoming = make_incoming(media_type="sticker", raw_text="[sticker]", file_id="file-1")
        state = make_state(incoming, should_respond=False)
        with (
            patch(DESCRIBE_STICKER_TARGET, new=AsyncMock()) as describe_sticker,
            patch(UPDATE_CONTENT_TARGET, new=AsyncMock()),
        ):
            result = await ingester(state)
        describe_sticker.assert_not_awaited()
        assert result["incoming"]["processed_text"] == "[sticker]"

    async def test_addressed_sticker_is_described(self, ingester):
        incoming = make_incoming(media_type="sticker", raw_text="[sticker]", file_id="file-1")
        state = make_state(incoming, should_respond=True)
        with (
            patch(DESCRIBE_STICKER_TARGET, new=AsyncMock(return_value="кот машет лапой")),
            patch(UPDATE_CONTENT_TARGET, new=AsyncMock()),
        ):
            result = await ingester(state)
        assert result["incoming"]["processed_text"] == "кот машет лапой"

    async def test_unknown_media_type_falls_back_to_raw_text(self, ingester):
        incoming = make_incoming(media_type="animation", raw_text="[animation]")
        state = make_state(incoming, should_respond=True)
        with patch(UPDATE_CONTENT_TARGET, new=AsyncMock()):
            result = await ingester(state)
        assert result["incoming"]["processed_text"] == "[animation]"
