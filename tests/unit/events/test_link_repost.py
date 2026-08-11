"""Caption composition and the 1024-character fitting ladder."""

from unittest.mock import AsyncMock, patch

from src.events.link_repost import (
    CAPTION_LIMIT,
    build_caption,
    fit_caption,
    truncate_at_sentence,
)

COMPRESS_PATCH_TARGET = "src.events.link_repost.compress_to_budget"


class TestBuildCaption:
    def test_credit_and_link_lead_the_caption(self):
        caption = build_caption("Про котиков.", "vasya", "https://youtu.be/abc")
        assert caption == "Скинул @vasya\nhttps://youtu.be/abc\n\nПро котиков."

    def test_summary_alone_when_the_original_survives(self):
        assert build_caption("Про котиков.", None, None) == "Про котиков."

    def test_summary_alone_when_only_one_of_the_pair_is_given(self):
        assert build_caption("Про котиков.", "vasya", None) == "Про котиков."
        assert build_caption("Про котиков.", None, "https://youtu.be/abc") == "Про котиков."


class TestTruncateAtSentence:
    def test_short_text_is_untouched(self):
        assert truncate_at_sentence("Коротко.", 100) == "Коротко."

    def test_cuts_at_the_last_sentence_boundary(self):
        text = "Первое предложение. Второе предложение. Третье."
        assert truncate_at_sentence(text, 30) == "Первое предложение."

    def test_falls_back_to_ellipsis_without_a_boundary(self):
        result = truncate_at_sentence("а" * 100, 10)
        assert len(result) <= 10
        assert result.endswith("…")

    def test_non_positive_budget_yields_empty_text(self):
        assert truncate_at_sentence("что угодно", 0) == ""


class TestFitCaption:
    async def test_within_budget_skips_compression(self):
        compress = AsyncMock()
        with patch(COMPRESS_PATCH_TARGET, new=compress):
            caption = await fit_caption("Коротко.", "vasya", "https://youtu.be/abc")
        compress.assert_not_awaited()
        assert caption.endswith("Коротко.")

    async def test_overflow_is_compressed(self):
        long_summary = "а" * 1200
        with patch(COMPRESS_PATCH_TARGET, new=AsyncMock(return_value="Сжато.")):
            caption = await fit_caption(long_summary, "vasya", "https://youtu.be/abc")
        assert caption.endswith("Сжато.")
        assert len(caption) <= CAPTION_LIMIT

    async def test_compressor_gets_the_budget_minus_overhead(self):
        compress = AsyncMock(return_value="Сжато.")
        with patch(COMPRESS_PATCH_TARGET, new=compress):
            await fit_caption("а" * 1200, "vasya", "https://youtu.be/abc")
        budget = compress.await_args.args[1]
        assert 0 < budget < CAPTION_LIMIT

    async def test_still_over_budget_falls_back_to_truncation(self):
        with patch(COMPRESS_PATCH_TARGET, new=AsyncMock(return_value="б" * 1200)):
            caption = await fit_caption("а" * 1200, "vasya", "https://youtu.be/abc")
        assert len(caption) <= CAPTION_LIMIT

    async def test_overflow_without_credit_still_fits(self):
        with patch(COMPRESS_PATCH_TARGET, new=AsyncMock(return_value="б" * 1200)):
            caption = await fit_caption("а" * 1200, None, None)
        assert len(caption) <= CAPTION_LIMIT

    async def test_absurd_url_overhead_still_fits(self):
        with patch(COMPRESS_PATCH_TARGET, new=AsyncMock(return_value="Сжато.")):
            caption = await fit_caption(
                "Про котиков.", "vasya", "https://example.com/" + "a" * 1200
            )
        assert len(caption) <= CAPTION_LIMIT
