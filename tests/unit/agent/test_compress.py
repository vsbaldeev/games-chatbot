"""Caption compressor — shrinks an over-budget summary without cutting it."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.compress import compress_to_budget

INVOKE_PATCH_TARGET = "src.agent.compress.ainvoke_with_backoff"


def make_result(content: str) -> MagicMock:
    result = MagicMock()
    result.content = content
    return result


class TestCompressToBudget:
    async def test_returns_compressed_text(self):
        with patch(INVOKE_PATCH_TARGET,
                   new=AsyncMock(return_value=make_result("  коротко  "))):
            assert await compress_to_budget("длинный текст", 100) == "коротко"

    async def test_budget_is_named_in_the_prompt(self):
        invoke = AsyncMock(return_value=make_result("коротко"))
        with patch(INVOKE_PATCH_TARGET, new=invoke):
            await compress_to_budget("длинный текст", 250)
        messages = invoke.await_args.args[1]
        assert "250" in messages[0].content

    async def test_llm_failure_returns_the_original(self):
        with patch(INVOKE_PATCH_TARGET, new=AsyncMock(side_effect=RuntimeError("boom"))):
            assert await compress_to_budget("длинный текст", 100) == "длинный текст"

    async def test_empty_output_returns_the_original(self):
        with patch(INVOKE_PATCH_TARGET, new=AsyncMock(return_value=make_result("  "))):
            assert await compress_to_budget("длинный текст", 100) == "длинный текст"
