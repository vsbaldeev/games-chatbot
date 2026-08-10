"""Static assertions on prompt text — catches accidental reintroduction of
mandatory-comedy instructions removed 2026-08-07 (see
docs/superpowers/specs/2026-08-07-reduce-bot-absurdity-design.md).
"""

from src.config.prompts import RESPONSE_PROMPT, VISION_PROMPT


class TestVisionPromptComedyIsConditional:
    """Verify that vision prompt's mandatory-comedy instruction is conditional, not forced."""

    def test_vision_prompt_does_not_mandate_funniness(self):
        """Assert the old "обязательно подметь" mandatory-comedy instruction is removed."""
        assert "обязательно подметь" not in VISION_PROMPT

    def test_vision_prompt_still_allows_noting_something_funny(self):
        """Assert the prompt still allows noting funny details when genuinely present."""
        assert "смешн" in VISION_PROMPT

    def test_vision_prompt_enforces_strict_length(self):
        """Assert the 1–2 sentence limit is strictly stated in the prompt."""
        assert "1–2 предложения" in VISION_PROMPT or "1-2 предложения" in VISION_PROMPT


class TestResponsePromptGroundedHumor:
    """Verify that response prompt enforces grounded humor and bans clickbait tails."""

    def test_response_prompt_forbids_clickbait_tails(self):
        """Assert the prohibition on clickbait-style chat engagement questions is present."""
        assert "кто из вас" in RESPONSE_PROMPT.lower()

    def test_response_prompt_requires_grounded_jokes(self):
        """Assert the requirement for grounded jokes (зацепиться) is present."""
        assert "зацепиться" in RESPONSE_PROMPT


class TestResponsePromptHasNoBiography:
    """Verify the reply prompt never tells Жора to narrate his own life.

    Members read the invented biography as noise (2026-08-10 decision): the
    scheduled life posts were retired and the reply prompt's standing order to
    improvise village activities was replaced by a deflect-and-redirect rule.
    """

    def test_does_not_instruct_improvising_village_life(self):
        """Assert the improvise-something-rural instruction is gone."""
        assert "импровизируй что-то будничное деревенское" not in RESPONSE_PROMPT

    def test_forbids_volunteering_his_own_affairs(self):
        """Assert the prompt explicitly bans unprompted self-narration."""
        assert "Про свою жизнь" in RESPONSE_PROMPT
        assert "сам не рассказывай" in RESPONSE_PROMPT

    def test_keeps_the_village_voice(self):
        """Assert the character's manner survives the biography's removal."""
        assert "Жора" in RESPONSE_PROMPT
        assert "сарказм" in RESPONSE_PROMPT.lower()
