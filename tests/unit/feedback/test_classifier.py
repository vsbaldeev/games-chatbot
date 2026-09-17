"""Correction classifier: one small LLM call deciding whether a reply to a
bot message disputes a factual claim in it."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.feedback.classifier import classify_correction

INVOKE_TARGET = "src.feedback.classifier.ainvoke_with_backoff"


def make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    return response


class TestClassifyCorrection:
    async def test_correction_verdict_returns_true(self):
        with patch(INVOKE_TARGET, AsyncMock(return_value=make_response("CORRECTION"))):
            assert await classify_correction("Half-Life 3 вышел в 2024.", "враньё, его не выпускали") is True

    async def test_other_verdict_returns_false(self):
        with patch(INVOKE_TARGET, AsyncMock(return_value=make_response("OTHER"))):
            assert await classify_correction("Бот сказал что-то", "ору с тебя") is False

    async def test_unparseable_verdict_fails_closed_to_false(self):
        with patch(INVOKE_TARGET, AsyncMock(return_value=make_response("маловероятно"))):
            assert await classify_correction("x", "y") is False

    async def test_llm_error_fails_closed_to_false(self):
        with patch(INVOKE_TARGET, AsyncMock(side_effect=RuntimeError("quota"))):
            assert await classify_correction("x", "y") is False
