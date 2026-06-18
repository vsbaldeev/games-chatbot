"""Groq-specific LangChain agent middleware and async call guard."""

import re
from typing import Any, Callable

import groq
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.exceptions import ContextLengthError, DailyLimitError, RateLimitError

CONTEXT_LENGTH_PHRASES = (
    "context_length_exceeded",
    "request too large",
    "string_above_max_length",
    "maximum context length",
    "input too long",
    "tokens_in_context",
)

DAILY_LIMIT_PHRASES = ("per day", "daily", "tokens_per_day")


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from text.

    Args:
        text: Input string potentially containing think blocks.

    Returns:
        Stripped string with all think blocks removed and surrounding whitespace trimmed.
    """
    return ThinkingStripper.THINK_RE.sub("", text).strip()


def should_retry(err: Exception) -> bool:
    """Return ``True`` only for transient Groq rate limits (TPM), not daily quota (TPD).

    Args:
        err: Exception raised by the model call.

    Returns:
        ``True`` for transient 429s worth retrying; ``False`` otherwise.
    """
    if not isinstance(err, groq.RateLimitError):
        return False
    error_str = str(err).lower()
    return not any(phrase in error_str for phrase in DAILY_LIMIT_PHRASES)


async def guarded_ainvoke(runnable, *args, **kwargs) -> Any:
    """Call ``runnable.ainvoke`` and map Groq errors to typed pipeline exceptions.

    Args:
        runnable: Any object with an async ``ainvoke`` method.
        *args: Positional arguments forwarded to ``ainvoke``.
        **kwargs: Keyword arguments forwarded to ``ainvoke``.

    Returns:
        The result from ``ainvoke``.

    Raises:
        ContextLengthError: For 400 errors matching context-length phrases.
        DailyLimitError: For 429 errors matching daily-quota phrases.
        RateLimitError: For other 429 rate-limit errors.
    """
    try:
        return await runnable.ainvoke(*args, **kwargs)
    except groq.BadRequestError as err:
        if any(phrase in str(err).lower() for phrase in CONTEXT_LENGTH_PHRASES):
            raise ContextLengthError("Input exceeds model context window") from err
        raise
    except groq.RateLimitError as err:
        error_str = str(err).lower()
        if any(phrase in error_str for phrase in DAILY_LIMIT_PHRASES):
            raise DailyLimitError("Daily token quota exhausted") from err
        raise RateLimitError("Rate limit exhausted") from err


class ToolMessageSanitizer(AgentMiddleware):
    """Replace empty ToolMessage content with a placeholder before each model call.

    Groq rejects tool messages with empty or missing content (HTTP 400).
    """

    async def abefore_model(self, state, runtime):
        """Replace empty ToolMessage content with ``(no output)`` before model call.

        Args:
            state: Agent state dict containing the ``messages`` list.
            runtime: Agent runtime (unused).

        Returns:
            Updated state dict if any messages were sanitized, otherwise ``None``.
        """
        messages = state["messages"]
        changed = False
        sanitized = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and not (msg.content or "").strip():
                sanitized.append(ToolMessage(
                    content="(no output)",
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                ))
                changed = True
            else:
                sanitized.append(msg)
        return {"messages": sanitized} if changed else None


class ThinkingStripper(AgentMiddleware):
    """Strip ``<think>...</think>`` blocks from AI message content after each model call.

    Reasoning models (e.g. Qwen3) prepend internal reasoning traces before the
    answer.  Stripping at the executor level keeps the message history clean for
    subsequent tool decisions and prevents traces from leaking to callers.
    """

    THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

    @staticmethod
    def strip_message(msg):
        """Strip think blocks from a single message, returning a new instance if changed.

        Intended for use in LCEL chains as a post-processing step.

        Args:
            msg: Any message object; non-string or missing content is returned unchanged.

        Returns:
            A new message with stripped content if a think block was found,
            otherwise the original message unchanged.
        """
        if not isinstance(getattr(msg, "content", None), str):
            return msg
        stripped = ThinkingStripper.THINK_RE.sub("", msg.content).strip()
        if stripped == msg.content:
            return msg
        return msg.model_copy(update={"content": stripped})

    async def aafter_model(self, state, runtime) -> dict | None:
        """Remove think blocks from the last AI message in the state.

        Args:
            state: Agent state dict containing the ``messages`` list.
            runtime: Agent runtime (unused).

        Returns:
            Updated state dict with stripped content if a think block was found,
            otherwise ``None``.
        """
        messages = state["messages"]
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage) or not isinstance(last.content, str):
            return None
        stripped = ThinkingStripper.THINK_RE.sub("", last.content).strip()
        if stripped == last.content:
            return None
        return {"messages": messages[:-1] + [last.model_copy(update={"content": stripped})]}


class GroqContextGuard(AgentMiddleware):
    """Convert Groq 400 context-window errors to ``ContextLengthError``.

    Placed inside ``ModelFallbackMiddleware`` so each per-model call raises
    ``ContextLengthError`` instead of the raw ``groq.BadRequestError``.

    Only overrides the async path (``awrap_model_call``). The worker executor
    is always invoked asynchronously, so the sync path is never reached.
    """

    async def awrap_model_call(self, request, handler: Callable) -> Any:
        """Intercept model call and reclassify context-length failures.

        Args:
            request: Model request forwarded to the handler unchanged.
            handler: Async callable that executes the underlying model.

        Returns:
            Model response on success.

        Raises:
            ContextLengthError: When the request exceeds the model's context window.
        """
        try:
            return await handler(request)
        except groq.BadRequestError as err:
            if any(phrase in str(err).lower() for phrase in CONTEXT_LENGTH_PHRASES):
                raise ContextLengthError("Input exceeds model context window") from err
            raise
