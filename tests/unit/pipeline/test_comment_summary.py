"""Comment summary — short in-character audience reaction, never verbatim quotes."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline import comment_summary

INVOKE_PATCH_TARGET = "src.pipeline.comment_summary.ainvoke_with_backoff"


def make_result(content: str, finish_reason: str = "stop") -> MagicMock:
    result = MagicMock()
    result.content = content
    result.response_metadata = {"finish_reason": finish_reason}
    return result


class TestSummarizeComments:
    async def test_no_comments_skips_model_call(self):
        with patch(INVOKE_PATCH_TARGET, new=AsyncMock()) as mock_invoke:
            result = await comment_summary.summarize_comments([])
        assert result == ""
        mock_invoke.assert_not_awaited()

    async def test_none_comments_skips_model_call(self):
        with patch(INVOKE_PATCH_TARGET, new=AsyncMock()) as mock_invoke:
            result = await comment_summary.summarize_comments(None)
        assert result == ""
        mock_invoke.assert_not_awaited()

    async def test_comments_with_only_blank_text_skips_model_call(self):
        with patch(INVOKE_PATCH_TARGET, new=AsyncMock()) as mock_invoke:
            result = await comment_summary.summarize_comments([{"text": "   ", "like_count": 5}])
        assert result == ""
        mock_invoke.assert_not_awaited()

    async def test_successful_summary_is_labelled(self):
        with patch(INVOKE_PATCH_TARGET, new=AsyncMock(return_value=make_result("всем зашло, хвалят монтаж"))):
            result = await comment_summary.summarize_comments(
                [{"text": "огонь ролик", "like_count": 40}]
            )
        assert result == "[Реакция комментаторов]:\nвсем зашло, хвалят монтаж"

    async def test_comments_are_capped_and_truncated_in_the_prompt(self):
        many_comments = [
            {"text": f"комментарий номер {index} " + "x" * 250, "like_count": index}
            for index in range(comment_summary.MAX_COMMENTS + 5)
        ]
        invoke = AsyncMock(return_value=make_result("сводка"))
        with patch(INVOKE_PATCH_TARGET, new=invoke):
            await comment_summary.summarize_comments(many_comments)
        sent = invoke.await_args.args[1][1].content
        assert sent.count("комментарий номер") == comment_summary.MAX_COMMENTS
        for line in sent.splitlines():
            assert len(line) <= comment_summary.COMMENT_CHAR_LIMIT + 20

    async def test_model_failure_returns_empty(self):
        with patch(INVOKE_PATCH_TARGET, new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await comment_summary.summarize_comments([{"text": "x", "like_count": 1}])
        assert result == ""

    async def test_empty_model_output_returns_empty(self):
        with patch(INVOKE_PATCH_TARGET, new=AsyncMock(return_value=make_result("   "))):
            result = await comment_summary.summarize_comments([{"text": "x", "like_count": 1}])
        assert result == ""

    async def test_finish_reason_length_is_logged_but_result_still_used(self, caplog):
        with patch(INVOKE_PATCH_TARGET, new=AsyncMock(return_value=make_result("сводка", "length"))):
            result = await comment_summary.summarize_comments([{"text": "x", "like_count": 1}])
        assert result == "[Реакция комментаторов]:\nсводка"
