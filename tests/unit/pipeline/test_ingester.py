"""MessageIngester tests — text-message path only (Shorts + social-link triggers).

Characterizes the existing Shorts behavior first (no test file covered this
before), then adds the new social-link behavior alongside it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipeline.ingester import (
    MessageIngester,
    is_low_confidence_transcript,
    summarize_social_link,
    transcribe_bytes,
)
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


class TestSummarizeSocialLink:
    """Exercises summarize_social_link itself (not mocked out), per handler."""

    async def test_unknown_handler_name_returns_empty(self):
        content_block, video_bytes = await summarize_social_link("nonexistent_handler", "url")
        assert (content_block, video_bytes) == ("", None)

    async def test_handler_fetch_raising_degrades_to_empty(self):
        broken_handler = MagicMock()
        broken_handler.name = "youtube_video"
        broken_handler.fetch = AsyncMock(side_effect=ValueError("malformed payload shape"))
        with patch(HANDLERS_TARGET, [broken_handler]):
            content_block, video_bytes = await summarize_social_link(
                "youtube_video", "https://www.youtube.com/watch?v=abc"
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


class TestLowConfidenceTranscript:
    def test_empty_segments_is_not_low_confidence(self):
        assert is_low_confidence_transcript([]) is False

    def test_high_confidence_segments_pass(self):
        segments = [{"avg_logprob": -0.2}, {"avg_logprob": -0.3}]
        assert is_low_confidence_transcript(segments) is False

    def test_low_mean_confidence_is_flagged(self):
        segments = [{"avg_logprob": -0.2}, {"avg_logprob": -1.5}]
        assert is_low_confidence_transcript(segments) is True

    def test_uniformly_low_confidence_is_flagged(self):
        segments = [{"avg_logprob": -0.9}, {"avg_logprob": -0.95}]
        assert is_low_confidence_transcript(segments) is True


class TestTranscribeBytesConfidenceFlag:
    async def test_clean_transcript_returns_not_low_confidence(self):
        mock_result = MagicMock()
        mock_result.text = "четкая речь без шума"
        mock_result.segments = [{"avg_logprob": -0.2}]
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_result)
        with patch("src.pipeline.ingester.AsyncGroq", return_value=mock_client):
            text, low_confidence = await transcribe_bytes(b"fake audio", "voice")
        assert text == "четкая речь без шума"
        assert low_confidence is False

    async def test_noisy_transcript_is_flagged_not_dropped(self):
        mock_result = MagicMock()
        mock_result.text = "может не только до попадает хер пойми"
        mock_result.segments = [{"avg_logprob": -0.2}, {"avg_logprob": -1.5}]
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_result)
        with patch("src.pipeline.ingester.AsyncGroq", return_value=mock_client):
            text, low_confidence = await transcribe_bytes(b"fake audio", "voice")
        assert text == "может не только до попадает хер пойми"
        assert low_confidence is True
