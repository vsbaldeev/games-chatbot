"""Transcription tests: confidence flags on Whisper segments."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline.ingest.transcription import is_low_confidence_transcript, transcribe_bytes


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
        with patch("src.pipeline.ingest.transcription.AsyncGroq", return_value=mock_client):
            text, low_confidence = await transcribe_bytes(b"fake audio", "voice")
        assert text == "четкая речь без шума"
        assert low_confidence is False

    async def test_noisy_transcript_is_flagged_not_dropped(self):
        mock_result = MagicMock()
        mock_result.text = "может не только до попадает хер пойми"
        mock_result.segments = [{"avg_logprob": -0.2}, {"avg_logprob": -1.5}]
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_result)
        with patch("src.pipeline.ingest.transcription.AsyncGroq", return_value=mock_client):
            text, low_confidence = await transcribe_bytes(b"fake audio", "voice")
        assert text == "может не только до попадает хер пойми"
        assert low_confidence is True
