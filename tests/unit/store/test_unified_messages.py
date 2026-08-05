"""
Photo content lifecycle tests (pure functions, no DB or LLM).

Scenario from fe3efb6: photos whose stored content was a caption rather than
the bare placeholder were treated as already-described.  The vision LLM was
never called, so "what's in this picture?" got "there is no picture" back.

The fix introduced an explicit placeholder-prefix convention: any content that
starts with "[photo]" still needs a vision description, regardless of whether
a caption follows.
"""

import pytest

from src.store.unified_messages import (
    PHOTO_PLACEHOLDER,
    combine_description_and_caption,
    display_photo_content,
    extract_photo_caption,
    format_photo_content,
    needs_photo_description,
)


class TestFormatPhotoContent:
    def test_caption_produces_placeholder_prefix_plus_caption(self):
        result = format_photo_content("Reddit post title")
        assert result == f"{PHOTO_PLACEHOLDER}\nReddit post title"

    def test_no_caption_produces_bare_placeholder(self):
        result = format_photo_content(None)
        assert result == PHOTO_PLACEHOLDER

    def test_empty_string_caption_produces_bare_placeholder(self):
        result = format_photo_content("")
        assert result == PHOTO_PLACEHOLDER


class TestNeedsPhotoDescription:
    """The bug: only the bare placeholder was checked; captioned rows slipped through."""

    def test_bare_placeholder_needs_description(self):
        assert needs_photo_description(PHOTO_PLACEHOLDER) is True

    def test_placeholder_with_caption_needs_description(self):
        # Regression (fe3efb6): this case was returning False before the fix.
        content = f"{PHOTO_PLACEHOLDER}\nReddit post title here"
        assert needs_photo_description(content) is True

    def test_vision_description_does_not_need_description(self):
        assert needs_photo_description("A cat sitting on a windowsill.") is False

    def test_description_with_caption_suffix_does_not_need_description(self):
        content = "A cat sitting on a windowsill.\n(подпись: cute cat)"
        assert needs_photo_description(content) is False


class TestExtractPhotoCaption:
    def test_extracts_caption_from_placeholder_prefixed_content(self):
        content = f"{PHOTO_PLACEHOLDER}\nmy caption text"
        assert extract_photo_caption(content) == "my caption text"

    def test_bare_placeholder_returns_empty_string(self):
        assert extract_photo_caption(PHOTO_PLACEHOLDER) == ""

    def test_plain_description_returns_empty_string(self):
        assert extract_photo_caption("some vision description") == ""


class TestCombineDescriptionAndCaption:
    def test_combines_description_with_caption(self):
        result = combine_description_and_caption("A cute cat.", "my cat photo")
        assert result == "A cute cat.\n(подпись: my cat photo)"

    def test_description_alone_when_caption_is_empty(self):
        result = combine_description_and_caption("A cute cat.", "")
        assert result == "A cute cat."


class TestDisplayPhotoContent:
    def test_strips_placeholder_prefix_from_captioned_form(self):
        content = f"{PHOTO_PLACEHOLDER}\nsome caption"
        assert display_photo_content(content) == "some caption"

    def test_bare_placeholder_returns_empty_string(self):
        assert display_photo_content(PHOTO_PLACEHOLDER) == ""

    def test_enriched_description_returned_unchanged(self):
        assert display_photo_content("A scenic mountain view.") == "A scenic mountain view."
