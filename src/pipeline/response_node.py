"""ResponseNode — personality LLM that turns worker facts into a chat reply."""

import logging
import time

from langchain_core.messages import AIMessage, HumanMessage

from src import config, log
from src.llm import needs_russian_correction, normalize_homoglyphs
from src.pipeline.reply_text import strip_markdown, strip_speaker_prefix, strip_writing_tics
from src.pipeline.response_prompt import (
    MEDIA_TRIGGER_LABELS,
    build_response_input,
    neutralize_speaker_lines,
    resolve_group_profile_directive,
    resolve_meme_directive,
)
from src.pipeline.state import BotState
from src.store import thread_history

logger = log.get_logger(__name__)


def build_past_messages(history: list[dict]) -> list[HumanMessage | AIMessage]:
    """Convert thread-history records into LangChain message objects.

    Args:
        history: List of dicts with ``role`` (``"human"`` or ``"ai"``) and
            ``content`` keys, ordered oldest-first.

    Returns:
        List of ``HumanMessage`` and ``AIMessage`` instances.
    """
    result: list[HumanMessage | AIMessage] = []
    for entry in history:
        if entry["role"] == "human":
            result.append(HumanMessage(content=entry["content"]))
        else:
            result.append(AIMessage(content=entry["content"]))
    return result


def serialize_history(past_messages: list[HumanMessage | AIMessage]) -> list[dict]:
    """Convert thread-history LangChain messages into JSON-serializable dicts.

    Used to persist the exact prompt a response LLM call saw — see
    src.store.llm_log.insert_call.

    Args:
        past_messages: History turns as built by build_past_messages, oldest-first.

    Returns:
        [{"role": "human"|"ai", "content": str}, ...], oldest-first.
    """
    return [
        {"role": "human" if isinstance(message, HumanMessage) else "ai", "content": message.content}
        for message in past_messages
    ]


async def persist_thread_turn(state: BotState, response_text: str) -> None:
    """Append the finished exchange to thread history.

    Must be called with the reply the chat actually saw — after language
    correction when it runs. Flat (non-reply) exchanges are stored under the
    prospective chain root (the triggering message id), so a follow-up reply
    to the bot's answer derives a thread pre-seeded with this exchange;
    reply-chain exchanges keep their derived thread id.

    Args:
        state: Current pipeline state.
        response_text: Final reply text as sent to the chat.
    """
    if not response_text.strip():
        return
    msg = state["incoming"]
    if state.get("is_flat_thread"):
        thread_id = thread_history.thread_id_for_root(msg["chat_id"], msg["message_id"])
    else:
        thread_id = state.get("thread_id") or str(msg["chat_id"])
    user_input = neutralize_speaker_lines(msg["processed_text"] or msg["raw_text"] or "")
    media_label = MEDIA_TRIGGER_LABELS.get(msg["media_type"])
    speaker = f"@{msg['username']}"
    human_content = (
        f"{speaker} [{media_label}]: {user_input}" if media_label
        else f"{speaker}: {user_input}"
    )
    await thread_history.append_turn(
        thread_id=thread_id,
        chat_id=msg["chat_id"],
        human_content=human_content,
        ai_content=strip_writing_tics(strip_speaker_prefix(strip_markdown(response_text))),
    )


def log_response_input(past_messages: list, enriched: str) -> None:
    """Dump the exact input the response LLM is about to see, at DEBUG level.

    Absurd replies are usually caused by something in the assembled prompt
    (a stale fact, a poisoned thread turn, a mis-framed trigger line), and
    that assembly is ephemeral — without this dump there is no way to see
    it after the fact. Thread history turns are logged as one-line excerpts
    (their full text lives in the ``thread_history`` table); the enriched
    final turn is logged verbatim because it exists nowhere else.

    Args:
        past_messages: Thread-history turns preceding the final human turn.
        enriched: The assembled final human turn passed to the LLM.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for position, message in enumerate(past_messages, start=1):
        speaker = "human" if isinstance(message, HumanMessage) else "ai"
        logger.debug(
            "Response LLM history %d/%d (%s): %s",
            position, len(past_messages), speaker, log.snippet(message.content),
        )
    logger.debug("Response LLM final turn:\n%s", enriched)


def log_response_usage(usage: dict) -> None:
    """Log the response call's real token usage, at DEBUG level.

    Reads ``AIMessage.usage_metadata`` — the serving model's own exact count
    (Groq's gpt-oss-120b/qwen3.8-27b, or the OpenRouter gemma fallback,
    whichever answered), captured post-call with no separate tokenizer and no
    third-party network call: a local per-block *estimate* was considered for
    this (tiktoken, and LangChain's own ``get_num_tokens()``) and rejected —
    both need a one-time download of tokenizer files from a public CDN
    (OpenAI's blob storage, or Hugging Face Hub for the GPT-2 fallback), and
    this bot must not make any such request. What Groq already returns for
    free, from the call already being made, is what gets logged instead.

    Args:
        usage: ``AIMessage.usage_metadata`` from the response call, or an
            empty dict when the model returned none.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if not usage:
        return
    logger.debug(
        "Response LLM token usage: input=%s output=%s total=%s",
        usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens"),
    )


def build_response_trace(
    past_messages: list, enriched: str, response_text: str, usage: dict, latency_ms: int
) -> dict:
    """Capture the exact prompt and response of one response-LLM call.

    Persisted by ``deliver_and_record`` (``src/events/messages.py``); the
    language-correction node rewrites ``response`` and ``language_corrected``
    when it retries.
    """
    return {
        "history_messages": serialize_history(past_messages),
        "user_prompt": enriched,
        "raw_response": response_text,
        "model": usage.get("model_name"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "latency_ms": latency_ms,
        "response": response_text,
        "language_corrected": False,
    }


class ResponseNode:
    """Generates the final personality-driven reply from gathered worker facts."""

    def __init__(self, agent) -> None:
        """Initialize ResponseNode.

        Args:
            agent: Agent instance used to invoke the response LLM.
        """
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        """Generate a response and persist the turn to thread history.

        Args:
            state: Current pipeline state with incoming message, context, and
                optional worker output.

        Returns:
            Dict with ``response`` and ``response_messages`` keys — the latter
            carries the assembled LangChain message list for the correction
            node — plus ``response_trace``, the exact prompt/response of this
            call for persistence by ``deliver_and_record``
            (``src/events/messages.py``), or None when no response LLM call
            ran (meme/group-profile requests).
        """
        if state.get("meme_request") or state.get("group_profile_request"):
            # The meme (or the group-profile message) is the whole reply: a
            # filler line here would be redundant, and a generated comment on
            # content this model never sees is the stacking failure the
            # 2026-08-07 absurdity work removed. Returning empty also skips
            # the response LLM entirely.
            return {"response": "", "response_messages": [], "response_trace": None}

        past_messages = await self.__load_history(state)
        enriched = self.__build_prompt(state, past_messages)
        messages = past_messages + [HumanMessage(content=enriched)]
        log_response_input(past_messages, enriched)
        usage: dict = {}
        started_at = time.monotonic()
        response_text = normalize_homoglyphs(await self.__generate(messages, usage))
        latency_ms = int((time.monotonic() - started_at) * 1000)
        log_response_usage(usage)
        response_trace = build_response_trace(past_messages, enriched, response_text, usage, latency_ms)

        # Foreign-script responses are persisted by LanguageCorrectionNode
        # after the retry, so history stores the reply the chat actually saw.
        if not needs_russian_correction(response_text or ""):
            await persist_thread_turn(state, response_text)

        return {"response": response_text, "response_messages": messages, "response_trace": response_trace}

    async def __load_history(self, state: BotState) -> list[HumanMessage | AIMessage]:
        """Load this thread's past turns as LangChain messages.

        Flat mentions are answered from recent chat context; the old flat
        bucket held only the bot's own stale exchanges.
        """
        if state.get("is_flat_thread"):
            return []
        thread_id = state.get("thread_id") or str(state["incoming"]["chat_id"])
        history = await thread_history.get_history(
            thread_id=thread_id, limit=config.MAX_HISTORY_MESSAGES
        )
        return build_past_messages(history)

    def __build_prompt(self, state: BotState, past_messages: list) -> str:
        """Assemble the enriched user turn for this run's state."""
        msg = state["incoming"]
        return build_response_input(
            msg["username"],
            msg["processed_text"] or msg["raw_text"] or "",
            state.get("worker_output") or "",
            state.get("context"),
            state.get("response_trigger") or "explicit",
            has_thread_history=bool(past_messages),
            media_type=msg["media_type"],
            is_bot_insult=bool(state.get("is_bot_insult")),
            wind_down=bool(state.get("wind_down")),
            worker_tools_used=bool(state.get("worker_tools_used")),
            meme_directive=resolve_meme_directive(state),
            group_profile_directive=resolve_group_profile_directive(state),
            voice_low_confidence=bool(state.get("voice_low_confidence")),
        )

    async def __generate(self, messages: list, usage_sink: dict) -> str:
        """Delegate to the response agent.

        Args:
            messages: Assembled message list (history + human turn, no system prompt —
                the executor prepends it internally).
            usage_sink: Filled in-place with the call's ``usage_metadata`` (input/
                output/total tokens) when the model returned one, for the DEBUG
                logging in :func:`log_response_usage`.

        Returns:
            Reply text from the agent. Empty string if the agent returned nothing.

        Raises:
            ContextLengthError: If the prompt exceeds the model's context window.
            DailyLimitError: If all models have exhausted their daily token quota.
            RateLimitError: If rate-limit retries are exhausted on all models.
        """
        return await self.__agent.invoke_response(messages, usage_sink=usage_sink)
