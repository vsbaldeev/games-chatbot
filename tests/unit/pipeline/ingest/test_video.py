"""Video frame tests."""

import logging
from unittest.mock import AsyncMock, patch

from src.pipeline.ingest.video import extract_and_describe_frames
from tests.builders import make_rate_limit_error


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
            patch("src.pipeline.ingest.video.extract_frames_sync", return_value=[b"frame bytes"]),
            patch("src.pipeline.ingest.video.describe_frame", new=AsyncMock(side_effect=rate_limit_error)),
            caplog.at_level(logging.WARNING, logger="src.pipeline.ingest.video"),
        ):
            frame_results = await extract_and_describe_frames(b"irrelevant video bytes")
        assert frame_results == []
        assert "Frame description failed" in caplog.text
