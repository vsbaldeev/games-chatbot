"""Static assertions on prompt text — catches accidental reintroduction of
mandatory-comedy instructions removed 2026-08-07 (see
docs/superpowers/specs/2026-08-07-reduce-bot-absurdity-design.md).
"""

from src.config.prompts import RESPONSE_PROMPT, VISION_PROMPT


class TestVisionPromptComedyIsConditional:
    def test_vision_prompt_does_not_mandate_funniness(self):
        assert "обязательно подметь" not in VISION_PROMPT

    def test_vision_prompt_still_allows_noting_something_funny(self):
        assert "смешн" in VISION_PROMPT

    def test_vision_prompt_enforces_strict_length(self):
        assert "1–2 предложения" in VISION_PROMPT or "1-2 предложения" in VISION_PROMPT
