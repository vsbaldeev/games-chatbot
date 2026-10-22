"""check_media tests: transcribed media passes, empty media gets a canned reply."""

from src.pipeline.ingest.filter.media_check import (
    TRANSCRIPTION_FAILED_REPLIES,
    VISION_FAILED_REPLIES,
    check_media,
)
from src.pipeline.state import ContentKind
from tests.builders import make_incoming, make_state


class TestMediaCheck:
    async def test_no_transcription_gets_canned_failure_reply(self):
        """An addressed voice note with no transcript gets an honest canned
        reply instead of a pass-through that would hallucinate a reaction."""
        state = make_state(
            make_incoming(media_type="voice", processed_text=None),
            should_respond=True,
            response_trigger="explicit",
        )
        result = await check_media(state)
        assert result["should_respond"] is False
        assert result["response"] in TRANSCRIPTION_FAILED_REPLIES
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE

    async def test_no_vision_photo_is_a_media_failure_notice(self):
        state = make_state(
            make_incoming(media_type="photo", processed_text=None),
            should_respond=True,
            response_trigger="explicit",
        )
        result = await check_media(state)
        assert result["response"] in VISION_FAILED_REPLIES
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE

    async def test_whitespace_only_transcription_treated_as_missing(self):
        state = make_state(
            make_incoming(media_type="voice", processed_text="   "),
            should_respond=True,
            response_trigger="explicit",
        )
        result = await check_media(state)
        assert result["content_kind"] == ContentKind.MEDIA_FAILURE_NOTICE

    async def test_transcribed_media_is_classified_meaningful(self):
        state = make_state(
            make_incoming(media_type="voice", processed_text="расскажи про игру"),
            should_respond=True,
            response_trigger="explicit",
        )
        result = await check_media(state)
        assert result == {"filter_verdict": "MEANINGFUL"}
