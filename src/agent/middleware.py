"""Groq-specific LangChain agent middleware and async call guard."""

import asyncio
import random
import re
from typing import Any, Callable
from urllib.parse import urlparse

import groq
import openai
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from src import log
from src.agent.exceptions import ContextLengthError, DailyLimitError, RateLimitError

logger = log.get_logger(__name__)

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


RATE_LIMIT_ERRORS = (groq.RateLimitError, openai.RateLimitError)


def should_retry(err: Exception) -> bool:
    """Return ``True`` only for transient rate limits (TPM), not daily quota (TPD).

    Covers both Groq and OpenAI-compatible (OpenRouter) rate-limit errors, since
    the response/roast fallback chains mix both providers.

    Args:
        err: Exception raised by the model call.

    Returns:
        ``True`` for transient 429s worth retrying; ``False`` otherwise.
    """
    if not isinstance(err, RATE_LIMIT_ERRORS):
        return False
    error_str = str(err).lower()
    return not any(phrase in error_str for phrase in DAILY_LIMIT_PHRASES)


RATE_LIMIT_MAX_ATTEMPTS = 4
RATE_LIMIT_BASE_DELAY = 1.0
RATE_LIMIT_MAX_DELAY = 30.0
RATE_LIMIT_JITTER = 1.0


def retry_after_seconds(err: groq.RateLimitError) -> float | None:
    """Extract Groq's suggested ``retry-after`` delay from a rate-limit error.

    Args:
        err: The Groq rate-limit error whose response headers may carry the hint.

    Returns:
        The server-suggested delay in seconds, or ``None`` when absent or unparseable.
    """
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


async def ainvoke_with_backoff(runnable, messages, *, max_attempts: int = RATE_LIMIT_MAX_ATTEMPTS) -> Any:
    """Invoke ``runnable.ainvoke``, retrying transient Groq TPM 429s with backoff.

    Honors Groq's ``retry-after`` header when present, otherwise falls back to
    exponential backoff. Jitter desynchronises concurrent callers so a burst of
    background tasks does not re-stampede the shared token bucket in lockstep.
    Daily-quota 429s and all non-rate-limit errors propagate immediately.

    Args:
        runnable: Any object exposing an async ``ainvoke`` method.
        messages: The message payload forwarded to ``ainvoke``.
        max_attempts: Total number of attempts before giving up.

    Returns:
        The result of the first successful ``ainvoke`` call.

    Raises:
        groq.RateLimitError: If every attempt is exhausted on transient limits.
        Exception: Any non-retryable error is re-raised unchanged.
    """
    for attempt in range(max_attempts):
        try:
            return await runnable.ainvoke(messages)
        except groq.RateLimitError as err:
            if not should_retry(err) or attempt == max_attempts - 1:
                raise
            delay = retry_after_seconds(err)
            if delay is None:
                delay = min(RATE_LIMIT_BASE_DELAY * (2 ** attempt), RATE_LIMIT_MAX_DELAY)
            await asyncio.sleep(delay + random.uniform(0, RATE_LIMIT_JITTER))


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
        RateLimitError: For other 429 rate-limit errors (Groq or OpenRouter).
    """
    try:
        return await runnable.ainvoke(*args, **kwargs)
    except (groq.BadRequestError, openai.BadRequestError) as err:
        if any(phrase in str(err).lower() for phrase in CONTEXT_LENGTH_PHRASES):
            raise ContextLengthError("Input exceeds model context window") from err
        raise
    except RATE_LIMIT_ERRORS as err:
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


def _model_name(request) -> str:
    """Best-effort model identifier for logging, e.g. ``google/gemma-4-31b-it:free``.

    Args:
        request: The ``ModelRequest`` whose resolved model to name.

    Returns:
        The model's ``model_name`` attribute (present on both ``ChatGroq`` and
        ``ChatOpenAI``), or its ``repr`` if that attribute is somehow absent.
    """
    return getattr(request.model, "model_name", None) or repr(request.model)


def _provider_name(request) -> str:
    """Best-effort provider host for logging, e.g. ``groq.com`` or ``openrouter.ai``.

    Distinguishes a Groq call from an OpenRouter one — both can carry a
    ``ChatOpenAI``-shaped model_name at a glance, so the model name alone
    does not say which provider (and which rate-limit pool) actually served
    or rejected the call.

    Args:
        request: The ``ModelRequest`` whose resolved model to identify.

    Returns:
        The hostname from the model's configured API base URL when one is
        set (true for every ``ChatOpenAI`` instance in this codebase, since
        they are always built with OPENROUTER_BASE_URL), ``"groq.com"`` for
        a ``ChatGroq`` on its default endpoint, or the model class name as a
        last resort.
    """
    model = request.model
    base_url = getattr(model, "openai_api_base", None) or getattr(model, "groq_api_base", None)
    if base_url:
        host = urlparse(base_url).hostname
        if host:
            return host
    return "groq.com" if type(model).__name__ == "ChatGroq" else type(model).__name__


def _message_text(message: Any) -> str:
    """Best-effort plain text for one message, for DEBUG-safe logging.

    Args:
        message: A LangChain message (input or output).

    Returns:
        Its string content when non-empty; otherwise, for an ``AIMessage``
        requesting tools, the tool names (content is often empty on a
        tool-calling turn); otherwise an empty string.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str) and content:
        return content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        names = ", ".join(call.get("name", "?") for call in tool_calls)
        return f"<tool_calls: {names}>"
    return content if isinstance(content, str) else str(content)


def _format_messages(messages: list) -> str:
    """Render a message list as one compact, DEBUG-safe log line.

    Args:
        messages: LangChain messages — a request's input or a response's output.

    Returns:
        ``"<empty>"`` for an empty list, otherwise ``"role: snippet | role: snippet"``
        with each message's text collapsed and truncated via :func:`log.snippet`.
    """
    if not messages:
        return "<empty>"
    return " | ".join(
        f"{getattr(message, 'type', type(message).__name__)}: {log.snippet(_message_text(message))}"
        for message in messages
    )


class ModelAttemptLogger(AgentMiddleware):
    """Log which provider/model served (or failed) each call, with input/output.

    Neither ``ModelFallbackMiddleware`` nor ``ModelRetryMiddleware`` log
    anything, so without this there is no way to tell from the logs which
    model in the chain — or even which provider — a given request hit, or
    what was actually sent/returned. Placed inside
    ``ModelFallbackMiddleware``/``ModelRetryMiddleware`` (so it sees every
    retry and every fallover) and outside ``GroqContextGuard`` (so it logs
    the raw provider exception, before that guard reclassifies it).
    """

    async def awrap_model_call(self, request, handler: Callable) -> Any:
        """Log the resolved provider/model and input, then the call's outcome.

        Args:
            request: Model request forwarded to the handler unchanged.
            handler: Async callable that executes the underlying model.

        Returns:
            Model response on success.

        Raises:
            Exception: Whatever the handler raised, unchanged, after logging it.
        """
        provider = _provider_name(request)
        model_name = _model_name(request)
        input_summary = _format_messages(request.messages)
        try:
            result = await handler(request)
        except Exception as err:
            logger.warning(
                "Model call failed: %s/%s (%s: %s) input=%s",
                provider, model_name, type(err).__name__, err, input_summary,
            )
            raise
        logger.debug(
            "Model call served by: %s/%s input=%s output=%s",
            provider, model_name, input_summary, _format_messages(result.result),
        )
        return result


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
