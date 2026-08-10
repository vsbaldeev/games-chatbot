"""Unit tests for the meme vision gate.

Covers:
  - detect_image_mime: magic-byte sniffing for the formats meme CDNs serve
  - parse_verdict:     validation of the model's 0-10 JSON score
  - score_meme:        the None-on-failure contract that keeps an outage from
                       reading as a verdict against the candidate
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memes import judge

JPEG = b"\xff\xd8\xff\xe0" + b"payload"
PNG = b"\x89PNG\r\n\x1a\n" + b"payload"
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 "
GIF = b"GIF89a" + b"payload"

INVOKE = "src.memes.judge.ainvoke_with_backoff"
MAKE_LLM = "src.memes.judge.make_judge_llm"


def llm_response(content: str) -> MagicMock:
    """Build a mock LLM response carrying raw ``.content`` text."""
    return MagicMock(content=content)


# ---------------------------------------------------------------------------
# detect_image_mime
# ---------------------------------------------------------------------------

class TestDetectImageMime:
    @pytest.mark.parametrize(
        "image_bytes, expected",
        [
            (JPEG, "image/jpeg"),
            (PNG, "image/png"),
            (WEBP, "image/webp"),
            (GIF, "image/gif"),
            (b"GIF87a...", "image/gif"),
        ],
        ids=["jpeg", "png", "webp", "gif89a", "gif87a"],
    )
    def test_recognises_known_formats(self, image_bytes, expected):
        assert judge.detect_image_mime(image_bytes) == expected

    @pytest.mark.parametrize(
        "image_bytes",
        [b"", b"not an image at all", b"RIFF\x00\x00\x00\x00AVI "],
        ids=["empty", "garbage", "riff-but-not-webp"],
    )
    def test_unrecognised_falls_back_to_jpeg(self, image_bytes):
        assert judge.detect_image_mime(image_bytes) == "image/jpeg"


# ---------------------------------------------------------------------------
# parse_verdict
# ---------------------------------------------------------------------------

class TestParseVerdict:
    @pytest.mark.parametrize("raw_score, expected", [(0, 0), (7, 7), (10, 10), (8.6, 8)],
                             ids=["floor", "mid", "ceiling", "float-truncated"])
    def test_accepts_scores_in_range(self, raw_score, expected):
        assert judge.parse_verdict({"score": raw_score}) == expected

    @pytest.mark.parametrize(
        "data",
        [
            None,
            {},
            {"verdict": 7},
            {"score": None},
            {"score": "7"},
            {"score": True},
            {"score": -1},
            {"score": 11},
        ],
        ids=["none-input", "empty", "wrong-key", "null-score", "string-score",
             "bool-score", "below-range", "above-range"],
    )
    def test_rejects_unusable_verdicts(self, data):
        assert judge.parse_verdict(data) is None


# ---------------------------------------------------------------------------
# score_meme
# ---------------------------------------------------------------------------

class TestScoreMeme:
    async def test_returns_parsed_score(self):
        with patch(MAKE_LLM, MagicMock()), \
             patch(INVOKE, AsyncMock(return_value=llm_response('{"score": 9}'))):
            assert await judge.score_meme(JPEG) == 9

    async def test_sends_image_with_sniffed_mime_type(self):
        with patch(MAKE_LLM, MagicMock()), \
             patch(INVOKE, AsyncMock(return_value=llm_response('{"score": 8}'))) as invoke:
            await judge.score_meme(PNG)
        image_part = invoke.await_args.args[1][1].content[0]
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    async def test_sends_no_caption_text_to_the_model(self):
        """The gate must judge the image alone — none will be sent with it."""
        with patch(MAKE_LLM, MagicMock()), \
             patch(INVOKE, AsyncMock(return_value=llm_response('{"score": 8}'))) as invoke:
            await judge.score_meme(JPEG)
        human_content = invoke.await_args.args[1][1].content
        assert [part["type"] for part in human_content] == ["image_url"]

    @pytest.mark.parametrize(
        "content",
        ["not json at all", "", '{"score": 42}', "[1, 2, 3]"],
        ids=["non-json", "empty", "out-of-range", "json-but-not-object"],
    )
    async def test_unusable_response_returns_none(self, content):
        with patch(MAKE_LLM, MagicMock()), \
             patch(INVOKE, AsyncMock(return_value=llm_response(content))):
            assert await judge.score_meme(JPEG) is None

    async def test_llm_failure_returns_none(self):
        with patch(MAKE_LLM, MagicMock()), \
             patch(INVOKE, AsyncMock(side_effect=RuntimeError("groq down"))):
            assert await judge.score_meme(JPEG) is None
