"""WorkerNode — agent that gathers facts using tools."""

import datetime
import logging

from langchain_core.callbacks import AsyncCallbackHandler

from src import log
from src.llm import ContextLengthError, DailyLimitError, RateLimitError
from src.pipeline.response_prompt import neutralize_speaker_lines, row_speaker, truncate_row_content
from src.pipeline.state import BotState, WorkerStatus
from src.store import unified_messages

logger = log.get_logger(__name__)

SEARCH_TOOLS = frozenset({"web_search"})
RECENT_FILL_LIMIT = 10


def worker_is_skipped(state: BotState) -> bool:
    """True when the reply needs no tool facts: personality-only replies and link summaries.

    A comeback needs personality, not IGDB lookups: skipping the worker avoids
    a «🔍 Ищу…» notification before the burn and junk search output steering
    it. Same for Shorts and social-link summaries — the source material is
    already in processed_text — and for wind-down brush-offs, which are one
    short phrase closing the conversation. A meme request answers with the
    image alone and no text at all, so gathered facts would have nowhere to
    go. Same for a group-profile request — it runs its own dedicated LLM call
    over the roster's own dossiers, not IGDB-style tool facts.
    """
    return bool(
        state.get("is_bot_insult")
        or state.get("wind_down")
        or state.get("meme_request")
        or state.get("group_profile_request")
        or state.get("response_trigger") in ("youtube_short", "social_link")
    )


def empty_worker_update(status: WorkerStatus) -> dict:
    """State update for every path that leaves the response without worker facts."""
    return {
        "worker_output": "",
        "search_notification_msg": None,
        "worker_tools_used": False,
        "worker_status": status,
    }


async def skip_worker(state: BotState) -> dict:
    """Graph node for replies that bypass the worker; records why no facts were gathered."""
    return empty_worker_update(WorkerStatus.SKIPPED)


def log_worker_call(worker_input: str, output: str, tools_used: bool) -> None:
    """Dump the worker's exact input and gathered output, at DEBUG level.

    The worker's output is the only trace of what it actually found (or
    fabricated); once it is folded into the response prompt there is no way
    to tell whether a wrong answer came from the worker or from the
    personality LLM without this dump.

    Args:
        worker_input: The exact prompt string sent to the worker agent.
        output: The worker's returned text, possibly empty.
        tools_used: Whether at least one tool actually ran.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("Worker input:\n%s", worker_input)
    logger.debug("Worker output (tools_used=%s): %s", tools_used, output or "(empty)")


class SearchNotificationCallback(AsyncCallbackHandler):
    """Sends a Telegram notification before web_search runs.

    The sent Message object is stored as ``sent_message`` so the caller can
    later edit it with the final response instead of sending a second message.
    """

    def __init__(self, message) -> None:
        """Initialize the callback.

        Args:
            message: Telegram Message object used to send the notification reply.
        """
        super().__init__()
        self.__message = message
        self.__notified = False
        self.sent_message = None

    async def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Send a search notification on the first web_search tool invocation.

        Args:
            serialized: Tool metadata dict containing at least a ``name`` key.
            input_str: Raw input string passed to the tool (unused).
            **kwargs: Additional keyword arguments forwarded by LangChain.
        """
        tool_name = serialized.get("name", "")
        if tool_name not in SEARCH_TOOLS or self.__notified:
            return
        self.__notified = True
        try:
            self.sent_message = await self.__message.reply_text("🔍 Ищу, подожди немного...")
        except Exception as err:
            logger.warning("Failed to send search notification: %s", err)


class WorkerNode:
    """Calls the worker agent and stores gathered facts in state."""

    def __init__(self, agent) -> None:
        """Initialize the worker node.

        Args:
            agent: Agent instance providing the worker executor and fallback logic.
        """
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        """Run the worker agent and return gathered facts.

        Absorbs ContextLengthError and unexpected exceptions by returning empty
        output so the pipeline can still attempt a response. DailyLimitError and
        RateLimitError are re-raised for the top-level handler to surface to the user.

        Args:
            state: Current pipeline state containing the incoming message and context.

        Returns:
            Dict with keys ``worker_output`` (str), ``search_notification_msg``
            (the sent Telegram Message or None) and ``worker_tools_used``
            (True when at least one tool actually ran; False on every error
            path) and ``worker_status`` (a WorkerStatus: OK when the worker
            ran, CONTEXT_OVERFLOW / FAILED when an error was absorbed and the
            response has no worker facts behind it). Replies that need no
            facts never reach this node; see ``skip_worker``.

        Raises:
            DailyLimitError: Propagated from ``agent.invoke_worker`` on daily quota exhaustion.
            RateLimitError: Propagated from ``agent.invoke_worker`` on transient rate limits.
        """
        msg = state["incoming"]
        worker_input = self.__build_worker_input(msg, state.get("context"))
        tg_message = msg["update"].message
        callback = SearchNotificationCallback(tg_message) if tg_message is not None else None
        callbacks = [callback] if callback is not None else None

        try:
            output, tools_used = await self.__agent.invoke_worker(worker_input, callbacks=callbacks)
            log_worker_call(worker_input, output, tools_used)
            notification_msg = callback.sent_message if callback is not None else None
            return {
                "worker_output": output,
                "search_notification_msg": notification_msg,
                "worker_tools_used": tools_used,
                "worker_status": WorkerStatus.OK,
            }
        except ContextLengthError as err:
            logger.warning("Worker context too long: %s", err)
            return empty_worker_update(WorkerStatus.CONTEXT_OVERFLOW)
        except (DailyLimitError, RateLimitError) as err:
            # Raising here skips this node's state patch entirely, so a
            # search notification already sent by the callback above would
            # otherwise be un-locatable by the top-level handler. Stash it on
            # the exception itself as a fallback route to the same message.
            if callback is not None and callback.sent_message is not None:
                err.search_notification_msg = callback.sent_message
            raise
        except Exception as err:
            logger.error("Worker failed: %s", err, exc_info=True)
            return empty_worker_update(WorkerStatus.FAILED)

    def __build_worker_input(self, msg: dict, context) -> str:
        """Assemble the prompt string sent to the worker agent.

        Includes the current UTC datetime, reply chain or recent history as
        context (mutually exclusive), and the user's question.

        Args:
            msg: IncomingMessage dict for the current update.
            context: Context dict with ``reply_chain`` and ``recent_history`` lists,
                or None if context is unavailable.

        Returns:
            Assembled prompt string ready to send to the worker executor.
        """
        user_input = msg["processed_text"] or msg["raw_text"] or ""
        username = msg["username"]
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        parts: list[str] = [f"Current datetime: {now}", ""]
        reply_chain = (context or {}).get("reply_chain") or []
        if reply_chain:
            parts.append("Context (reply chain):")
            for row in reply_chain:
                parts.append(self.__render_row(row))
            parts.append("")
        else:
            recent = ((context or {}).get("recent_history") or [])[:RECENT_FILL_LIMIT]
            if recent:
                parts.append("Recent chat context:")
                for row in reversed(recent):
                    parts.append(self.__render_row(row))
                parts.append("")
        parts.append(f"Question from @{username}: {neutralize_speaker_lines(user_input)}")
        return "\n".join(parts)

    @staticmethod
    def __render_row(row: dict) -> str:
        """Format a message row as ``speaker: content`` for the worker prompt.

        The bot's own past messages are labelled ``Ты (бот)`` rather than
        ``@username`` so the worker never treats them as another participant.
        Content is neutralized against forged speaker lines and capped, the
        same as the response prompt's rows — this prompt is flattened the same
        way and carries the same risk (see
        :func:`src.pipeline.response_prompt.neutralize_speaker_lines`).

        Args:
            row: Message row dict with ``user_id``, ``username``, ``content``,
                and ``media_type`` keys.

        Returns:
            Formatted string representation of the message.
        """
        content = unified_messages.display_media_content(row["media_type"], row["content"])
        content = truncate_row_content(neutralize_speaker_lines(content))
        return f"{row_speaker(row)}: {content}"
