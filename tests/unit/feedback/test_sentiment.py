"""Reaction-emoji sentiment: a fixed positive set, everything else negative.

Telegram sends some reactions with a trailing variation selector (U+FE0F) and
some without, so membership is tested on the normalized form — see
test_heart_with_variation_selector_is_positive, the case that motivated it.
"""

from src.feedback.sentiment import (
    POSITIVE_REACTIONS,
    is_positive_reaction,
    normalize_reaction,
)


class TestNormalizeReaction:
    def test_strips_variation_selector(self):
        assert normalize_reaction("❤️") == "❤"

    def test_leaves_bare_emoji_unchanged(self):
        assert normalize_reaction("\U0001f44d") == "\U0001f44d"


class TestIsPositiveReaction:
    def test_heart_with_variation_selector_is_positive(self):
        assert is_positive_reaction("❤️") is True

    def test_bare_heart_is_positive(self):
        assert is_positive_reaction("❤") is True

    def test_fire_is_positive(self):
        assert is_positive_reaction("\U0001f525") is True

    def test_thinking_face_is_negative(self):
        assert is_positive_reaction("\U0001f914") is False

    def test_poop_is_negative(self):
        assert is_positive_reaction("\U0001f4a9") is False

    def test_unknown_emoji_is_negative(self):
        assert is_positive_reaction("\U0001f996") is False

    def test_empty_string_is_negative(self):
        assert is_positive_reaction("") is False
