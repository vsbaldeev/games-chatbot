"""MessageIngester tests — text-message path only (Shorts + social-link triggers).

Characterizes the existing Shorts behavior first (no test file covered this
before), then adds the new social-link behavior alongside it.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipeline import shorts
from src.pipeline.ingester import (
    MessageIngester,
    extract_and_describe_frames,
    is_low_confidence_transcript,
    summarize_social_link,
    summarize_youtube_short,
    transcribe_bytes,
)
from tests.builders import make_incoming, make_rate_limit_error, make_state

SUMMARIZE_SHORT_TARGET = "src.pipeline.ingester.summarize_youtube_short"
SUMMARIZE_SOCIAL_LINK_TARGET = "src.pipeline.ingester.summarize_social_link"
UPDATE_CONTENT_TARGET = "src.pipeline.ingester.unified_messages.update_content"
HANDLERS_TARGET = "src.pipeline.ingester.social_links.HANDLERS"
DOWNLOAD_SHORT_TARGET = "src.pipeline.ingester.shorts.download_short"


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


class TestSummarizeYoutubeShort:
    """Exercises summarize_youtube_short itself (not mocked out).

    Mocks only the I/O boundary — the yt-dlp download and the Groq
    transcription client — the same shape TestSummarizeSocialLink uses for
    summarize_social_link. Frame extraction is left to run for real: on the
    fake video bytes used here it fails harmlessly (caught inside
    extract_and_describe_frames) and returns no frames, which is fine since
    these tests only care about the transcript half of the content block.
    """

    def __make_transcription_client(self, text: str) -> MagicMock:
        mock_result = MagicMock()
        mock_result.text = text
        mock_result.segments = [{"avg_logprob": -0.2}]
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_result)
        return mock_client

    async def test_content_block_is_a_plain_string_with_no_tuple_artifacts(self):
        info = {"title": "Тестовое видео", "channel": "TestChan", "duration": 15}
        mock_client = self.__make_transcription_client("привет всем, это тест")
        with (
            patch(DOWNLOAD_SHORT_TARGET, new=AsyncMock(return_value=(b"fake video bytes", info))),
            patch("src.pipeline.ingester.AsyncGroq", return_value=mock_client),
        ):
            content_block, video_bytes = await summarize_youtube_short("https://youtube.com/shorts/abc")
        assert isinstance(content_block, str)
        assert "(" not in content_block
        assert ")" not in content_block
        assert "'" not in content_block
        assert "привет всем, это тест" in content_block
        assert video_bytes == b"fake video bytes"

    async def test_transcript_over_char_limit_is_truncated(self):
        info = {"title": "Длинное видео"}
        long_transcript = "а" * (shorts.TRANSCRIPT_CHAR_LIMIT + 500)
        mock_client = self.__make_transcription_client(long_transcript)
        with (
            patch(DOWNLOAD_SHORT_TARGET, new=AsyncMock(return_value=(b"fake video bytes", info))),
            patch("src.pipeline.ingester.AsyncGroq", return_value=mock_client),
        ):
            content_block, _ = await summarize_youtube_short("https://youtube.com/shorts/abc")
        truncated = long_transcript[: shorts.TRANSCRIPT_CHAR_LIMIT] + "…"
        assert truncated in content_block
        assert long_transcript not in content_block

    async def test_no_transcript_and_no_frames_returns_empty(self):
        info = {"title": "Тишина"}
        mock_client = self.__make_transcription_client("")
        with (
            patch(DOWNLOAD_SHORT_TARGET, new=AsyncMock(return_value=(b"fake video bytes", info))),
            patch("src.pipeline.ingester.AsyncGroq", return_value=mock_client),
        ):
            content_block, video_bytes = await summarize_youtube_short("https://youtube.com/shorts/abc")
        assert (content_block, video_bytes) == ("", None)


class TestExtractAndDescribeFrames:
    """A daily-quota RateLimitError from the vision LLM (reproduced live against
    Groq: qwen/qwen3.6-27b hit its tokens-per-day cap) used to vanish inside
    ``asyncio.gather(..., return_exceptions=True)`` with no log line, so a
    Short that failed here looked identical in the logs to one where the
    frame just had nothing worth describing.
    """

    async def test_frame_description_failure_is_logged_and_degrades_to_no_frames(self, caplog):
        rate_limit_error = make_rate_limit_error(
            "Rate limit reached for model `qwen/qwen3.6-27b` ... tokens per day (TPD)"
        )
        with (
            patch("src.pipeline.ingester.extract_frames_sync", return_value=[b"frame bytes"]),
            patch("src.pipeline.ingester.describe_frame", new=AsyncMock(side_effect=rate_limit_error)),
            caplog.at_level(logging.WARNING, logger="src.pipeline.ingester"),
        ):
            frame_results = await extract_and_describe_frames(b"irrelevant video bytes")
        assert frame_results == []
        assert "Frame description failed" in caplog.text


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
