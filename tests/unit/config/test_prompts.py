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


class TestResponsePromptIsAnAssistant:
    """Verify the reply prompt presents an AI assistant, not a village character.

    Members disliked the Жора persona and read the bot as an assistant, so the
    character sheet, the village biography and the bot-and-Жора dual identity
    were removed (2026-10-13 decision).
    """

    def test_has_no_village_character(self):
        """Assert the village persona text is gone from the prompt."""
        assert "деревн" not in RESPONSE_PROMPT
        assert "глухой славянской" not in RESPONSE_PROMPT

    def test_forbids_inventing_a_biography(self):
        """Assert the prompt bans narrating a life the bot does not have."""
        assert "Никакой выдуманной биографии" in RESPONSE_PROMPT

    def test_admits_being_an_ai(self):
        """Assert the prompt states the bot is an AI assistant."""
        assert "ИИ-ассистент" in RESPONSE_PROMPT

    def test_keeps_the_name_as_an_address_handle(self):
        """Assert the name survives so that addressing the bot by it works."""
        assert "Жора" in RESPONSE_PROMPT
