"""Reaction-emoji sentiment: a fixed positive set, everything else negative.

Telegram sends some reactions with a trailing variation selector (U+FE0F) and
some without, so membership is tested on the normalized form — see
test_heart_with_variation_selector_is_positive, the case that motivated it.
"""

from src.feedback.sentiment import (
    POSITIVE_REACTIONS,
    is_positive_reaction,
    normalize_reaction,
    positive_reactions_sql_array,
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


class TestPositiveReactionsSqlArray:
    def test_contains_every_positive_reaction_normalized(self):
        array = positive_reactions_sql_array()
        assert set(array) == POSITIVE_REACTIONS
        assert all("️" not in emoji for emoji in array)

    def test_is_a_list_for_asyncpg_binding(self):
        assert isinstance(positive_reactions_sql_array(), list)


class TestRollupBinding:
    def test_rollup_sql_binds_the_positive_set_rather_than_inlining_it(self):
        """One definition of the emoji set: Python, passed as $3.

        An inlined literal in ROLLUP_SQL would drift from POSITIVE_REACTIONS
        the first time the operator edits one and not the other.
        """
        from src.store.feedback_metrics import ROLLUP_SQL

        assert "$3::text[]" in ROLLUP_SQL
        for emoji in POSITIVE_REACTIONS:
            assert emoji not in ROLLUP_SQL
