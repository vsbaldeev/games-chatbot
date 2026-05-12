"""Tests for agent utility functions: strip_thinking and apply_language_correction.

Regression for the Qwen3 thinking-block leak: the model emits <think>...</think>
in its raw output and the bot was forwarding that entire block to the user instead
of the actual answer.

Fix: strip_thinking() removes thinking blocks at every consumption point.
     apply_language_correction() checks only the visible answer for foreign script
     so thinking-internal foreign text doesn't trigger a spurious Russian retry.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent import apply_language_correction, strip_thinking


class TestStripThinking:
    def test_text_without_tags_returned_unchanged(self):
        assert strip_thinking("Через 191 день.") == "Через 191 день."

    def test_single_think_block_removed(self):
        text = "<think>some internal reasoning</think>Через 191 день."
        assert strip_thinking(text) == "Через 191 день."

    def test_multiline_think_block_removed(self):
        text = "<think>\nline one\nline two\n</think>Ответ."
        assert strip_thinking(text) == "Ответ."

    def test_only_think_block_gives_empty_string(self):
        assert strip_thinking("<think>just thinking</think>") == ""

    def test_empty_string_stays_empty(self):
        assert strip_thinking("") == ""

    def test_surrounding_whitespace_stripped(self):
        assert strip_thinking("<think>reasoning</think>  Ответ.  ") == "Ответ."

    def test_tag_matching_is_case_insensitive(self):
        assert strip_thinking("<THINK>reasoning</THINK>Результат.") == "Результат."


class TestApplyLanguageCorrection:
    async def test_clean_russian_answer_returns_message_unchanged(self):
        llm = MagicMock()
        ai_message = MagicMock()
        ai_message.content = "Это нормальный русский ответ."
        result = await apply_language_correction(llm, ai_message, [])
        assert result is ai_message
        llm.ainvoke.assert_not_called()

    async def test_foreign_script_inside_think_block_does_not_trigger_retry(self):
        """Reasoning steps may contain foreign words while thinking — only the
        visible answer outside the think block should trigger the language retry."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        ai_message = MagicMock()
        ai_message.content = "<think>こんにちは、考えています</think>Нормальный ответ."
        result = await apply_language_correction(llm, ai_message, [])
        assert result is ai_message
        llm.ainvoke.assert_not_called()

    async def test_foreign_script_in_visible_answer_triggers_retry(self):
        """When the actual answer outside the think block contains foreign script,
        the Russian-language retry must fire."""
        corrected = MagicMock()
        corrected.content = "Исправленный ответ."
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=corrected)
        ai_message = MagicMock()
        ai_message.content = "<think>normal thinking</think>こんにちは"
        result = await apply_language_correction(llm, ai_message, [])
        assert result is corrected
        llm.ainvoke.assert_called_once()
