"""Counter SQL for bot_message_feedback.

No database in the unit suite, so these pin the SQL each mutator sends:
which column moves, that a decrement never goes below zero, and that every
statement keeps `ignored` equal to "every counter is 0".
"""

import pytest

from src.store.message_feedback import (
    COUNTER_COLUMNS,
    REPLY_KIND_COLUMNS,
    decrement_sql,
    increment_sql,
    reaction_column,
)


class TestReactionColumn:
    def test_positive_emoji_goes_to_positive_reactions(self):
        assert reaction_column("🔥") == "positive_reactions"

    def test_heart_with_variation_selector_is_positive(self):
        assert reaction_column("❤️") == "positive_reactions"

    def test_other_emoji_goes_to_negative_reactions(self):
        assert reaction_column("🤡") == "negative_reactions"

    def test_custom_emoji_key_is_negative(self):
        assert reaction_column("custom:12345") == "negative_reactions"


class TestReplyKindColumns:
    def test_every_kind_maps_to_a_counter(self):
        assert set(REPLY_KIND_COLUMNS.values()) <= set(COUNTER_COLUMNS)

    def test_kinds_are_exactly_the_five_reply_outcomes(self):
        assert set(REPLY_KIND_COLUMNS) == {"positive", "negative", "neutral", "media", "unlabelled"}


class TestIncrementSql:
    def test_increments_the_named_column(self):
        assert "positive_replies = positive_replies + 1" in increment_sql("positive_replies")

    def test_any_increment_clears_ignored(self):
        assert "ignored = FALSE" in increment_sql("media_replies")

    def test_unknown_column_is_refused(self):
        with pytest.raises(ValueError):
            increment_sql("ignored; DROP TABLE x")


class TestDecrementSql:
    def test_floors_at_zero(self):
        assert "negative_reactions = GREATEST(negative_reactions - 1, 0)" in decrement_sql("negative_reactions")

    def test_recomputes_ignored_from_every_counter(self):
        sql = decrement_sql("positive_reactions")
        ignored_clause = sql.split("ignored =", 1)[1]
        for column in COUNTER_COLUMNS:
            if column != "positive_reactions":
                assert column in ignored_clause
        assert "GREATEST(positive_reactions - 1, 0)" in ignored_clause

    def test_unknown_column_is_refused(self):
        with pytest.raises(ValueError):
            decrement_sql("sent_at")
