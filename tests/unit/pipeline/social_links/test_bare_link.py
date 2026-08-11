"""Bare-link classification — the rule that decides whether the user's
message may be deleted. Fail-safe: anything left over counts as user text."""

from src.pipeline.shorts import SHORTS_URL_RE
from src.pipeline.social_links import HANDLERS, is_bare_link_message
from src.pipeline.social_links.instagram_reel import INSTAGRAM_URL_RE
from src.pipeline.social_links.youtube_video import YOUTUBE_VIDEO_URL_RE


class TestBareLinkMessage:
    def test_link_alone_is_bare(self):
        assert is_bare_link_message(
            "https://www.youtube.com/shorts/abc123", SHORTS_URL_RE
        )

    def test_link_with_tracking_params_is_bare(self):
        assert is_bare_link_message(
            "https://youtube.com/shorts/abc123?si=xYz", SHORTS_URL_RE
        )

    def test_surrounding_whitespace_is_bare(self):
        assert is_bare_link_message(
            "  https://www.instagram.com/reel/abc123/\n\n", INSTAGRAM_URL_RE
        )

    def test_short_youtu_be_form_is_bare(self):
        assert is_bare_link_message("https://youtu.be/abc123", YOUTUBE_VIDEO_URL_RE)

    def test_user_words_are_not_bare(self):
        assert not is_bare_link_message(
            "гляньте какая дичь https://www.youtube.com/shorts/abc123", SHORTS_URL_RE
        )

    def test_emoji_only_remainder_is_not_bare(self):
        assert not is_bare_link_message(
            "🔥 https://www.youtube.com/shorts/abc123", SHORTS_URL_RE
        )

    def test_second_link_remainder_is_not_bare(self):
        assert not is_bare_link_message(
            "https://www.youtube.com/shorts/abc123 https://example.com/x", SHORTS_URL_RE
        )

    def test_empty_text_is_not_bare(self):
        assert not is_bare_link_message(None, SHORTS_URL_RE)

    def test_emoji_glued_to_the_link_is_not_bare(self):
        assert not is_bare_link_message(
            "https://www.youtube.com/shorts/abc123🔥", SHORTS_URL_RE
        )

    def test_word_glued_to_the_link_is_not_bare(self):
        assert not is_bare_link_message(
            "https://www.youtube.com/shorts/abc123смотри", SHORTS_URL_RE
        )


class TestHandlersExposePattern:
    """Every handler's own pattern must match the links it claims."""

    def test_instagram_handler_pattern_matches_a_reel_link(self):
        handler = next(one for one in HANDLERS if one.name == "instagram_reel")
        assert handler.pattern.search("https://www.instagram.com/reel/abc123/")

    def test_youtube_handler_pattern_matches_a_watch_link(self):
        handler = next(one for one in HANDLERS if one.name == "youtube_video")
        assert handler.pattern.search("https://www.youtube.com/watch?v=abc123")

    def test_handler_pattern_ignores_the_other_platform(self):
        handler = next(one for one in HANDLERS if one.name == "instagram_reel")
        assert handler.pattern.search("https://www.youtube.com/watch?v=abc123") is None
