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
    AUDIO_PLACEHOLDER,
    CONTACT_PLACEHOLDER,
    DICE_PLACEHOLDER,
    DOCUMENT_PLACEHOLDER,
    LOCATION_PLACEHOLDER,
    PHOTO_PLACEHOLDER,
    VENUE_PLACEHOLDER,
    combine_description_and_caption,
    display_photo_content,
    extract_photo_caption,
    format_audio_content,
    format_contact_content,
    format_dice_content,
    format_document_content,
    format_location_content,
    format_photo_content,
    format_poll_content,
    format_venue_content,
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


class TestFormatDocumentContent:
    def test_file_name_is_appended_to_placeholder(self):
        assert format_document_content("report.pdf") == f"{DOCUMENT_PLACEHOLDER} report.pdf"

    def test_missing_file_name_produces_bare_placeholder(self):
        assert format_document_content(None) == DOCUMENT_PLACEHOLDER


class TestFormatAudioContent:
    def test_title_and_performer_take_priority(self):
        result = format_audio_content("track.mp3", "Komarovo", "Кино")
        assert result == f"{AUDIO_PLACEHOLDER} Кино - Komarovo"

    def test_title_alone_without_performer(self):
        result = format_audio_content("track.mp3", "Komarovo", None)
        assert result == f"{AUDIO_PLACEHOLDER} Komarovo"

    def test_falls_back_to_file_name_without_tags(self):
        result = format_audio_content("track.mp3", None, None)
        assert result == f"{AUDIO_PLACEHOLDER} track.mp3"

    def test_bare_placeholder_with_nothing_available(self):
        assert format_audio_content(None, None, None) == AUDIO_PLACEHOLDER


class TestFormatLocationContent:
    def test_coordinates_are_appended_to_placeholder(self):
        result = format_location_content(55.75, 37.61)
        assert result == f"{LOCATION_PLACEHOLDER} 55.75,37.61"


class TestFormatContactContent:
    def test_full_name_is_appended_to_placeholder(self):
        result = format_contact_content("Ivan", "Petrov")
        assert result == f"{CONTACT_PLACEHOLDER} Ivan Petrov"

    def test_missing_last_name_uses_first_name_only(self):
        result = format_contact_content("Ivan", None)
        assert result == f"{CONTACT_PLACEHOLDER} Ivan"

    def test_phone_number_is_never_included(self):
        result = format_contact_content("Ivan", "Petrov")
        assert "+" not in result


class TestFormatDiceContent:
    def test_emoji_and_value_are_appended_to_placeholder(self):
        result = format_dice_content("🎲", 4)
        assert result == f"{DICE_PLACEHOLDER} 🎲 4"


class TestFormatVenueContent:
    def test_title_and_address_are_appended_to_placeholder(self):
        result = format_venue_content("Кафе Пушкин", "Тверской бул. 26А")
        assert result == f"{VENUE_PLACEHOLDER} Кафе Пушкин, Тверской бул. 26А"


class TestFormatPollContent:
    def test_question_and_options_are_rendered_in_full(self):
        result = format_poll_content("Го в дуэль?", ["Да", "Нет", "Может быть"])
        assert result == "[опрос] Вопрос: Го в дуэль? Варианты: Да | Нет | Может быть"


class TestDisplayPhotoContent:
    def test_strips_placeholder_prefix_from_captioned_form(self):
        content = f"{PHOTO_PLACEHOLDER}\nsome caption"
        assert display_photo_content(content) == "some caption"

    def test_bare_placeholder_returns_empty_string(self):
        assert display_photo_content(PHOTO_PLACEHOLDER) == ""

    def test_enriched_description_returned_unchanged(self):
        assert display_photo_content("A scenic mountain view.") == "A scenic mountain view."
