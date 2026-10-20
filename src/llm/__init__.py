"""Shared LLM-call infrastructure: retry/backoff, agent middleware, errors, language fixes, vision factory."""

from src.llm.exceptions import ContextLengthError, DailyLimitError, RateLimitError
from src.llm.middleware import (
    CONTEXT_LENGTH_PHRASES,
    DAILY_LIMIT_PHRASES,
    GroqContextGuard,
    ModelAttemptLogger,
    ThinkingStripper,
    ToolMessageSanitizer,
    ainvoke_with_backoff,
    guarded_ainvoke,
    should_retry,
    strip_thinking,
)
from src.llm.language import (
    FOREIGN_SCRIPT_RE,
    apply_language_correction,
    needs_russian_correction,
    normalize_homoglyphs,
)
from src.llm.compress import compress_to_budget
from src.llm.vision import make_vision_llm

__all__ = [
    "ContextLengthError",
    "DailyLimitError",
    "RateLimitError",
    "CONTEXT_LENGTH_PHRASES",
    "DAILY_LIMIT_PHRASES",
    "GroqContextGuard",
    "ModelAttemptLogger",
    "ThinkingStripper",
    "ToolMessageSanitizer",
    "ainvoke_with_backoff",
    "guarded_ainvoke",
    "should_retry",
    "strip_thinking",
    "FOREIGN_SCRIPT_RE",
    "apply_language_correction",
    "needs_russian_correction",
    "normalize_homoglyphs",
    "compress_to_budget",
    "make_vision_llm",
]