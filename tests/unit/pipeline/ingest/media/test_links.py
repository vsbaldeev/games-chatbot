"""Link summarisation tests: Shorts and social-link handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline.ingest.links import shorts
from src.pipeline.ingest.media.links import summarize_social_link, summarize_youtube_short

HANDLERS_TARGET = "src.pipeline.ingest.media.links.social_links.HANDLERS"
DOWNLOAD_SHORT_TARGET = "src.pipeline.ingest.media.links.shorts.download_short"


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
            patch("src.pipeline.ingest.media.transcription.AsyncGroq", return_value=mock_client),
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
            patch("src.pipeline.ingest.media.transcription.AsyncGroq", return_value=mock_client),
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
            patch("src.pipeline.ingest.media.transcription.AsyncGroq", return_value=mock_client),
        ):
            content_block, video_bytes = await summarize_youtube_short("https://youtube.com/shorts/abc")
        assert (content_block, video_bytes) == ("", None)
