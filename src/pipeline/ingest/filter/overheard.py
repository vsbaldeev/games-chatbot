"""
OverheardNode — filter-stage node for messages that mention the bot by word.

Messages routed by the router's bot-word check (``response_trigger="insult_check"``)
are classified with the last few chat messages as context, so the model can tell
this bot from game bots, other Telegram bots and people playing «как бот». A
BOT_INSULT verdict acts only after a stronger model confirms it on the same
input (disagreement or a confirmation error resolves to silence); everything
else is dropped silently — no emoji reaction, because the bot was never
addressed and reacting would be noise.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src import config, log
from src.config.prompts import OVERHEARD_SYSTEM
from src.pipeline.ingest.filter.classify import ask_label, extract_passive_facts, make_filter_llm
from src.pipeline.state import BotState, Verdict
from src.store import unified_messages

logger = log.get_logger(__name__)

# Overheard bot-word checks see the last few chat messages so the classifier
# can resolve which bot (or person playing «как бот») is being talked about.
OVERHEARD_CONTEXT_LIMIT = 5
OVERHEARD_CONTEXT_CHAR_LIMIT = 120


def build_overheard_input(text: str, recent: list[dict]) -> str:
    """Assemble the human message for the overheard bot-word classifier.

    Args:
        text: The overheard message text under classification.
        recent: Recent chat rows (newest-first) providing referent context;
            may be empty, in which case the bare text is returned.

    Returns:
        The bare text when no context is available; otherwise the last few
        chat messages (oldest-first, truncated) followed by the message to
        classify, each section clearly labelled.
    """
    if not recent:
        return text
    lines = []
    for row in reversed(recent[:OVERHEARD_CONTEXT_LIMIT]):
        content = (row.get("content") or "").strip()
        if len(content) > OVERHEARD_CONTEXT_CHAR_LIMIT:
            content = content[:OVERHEARD_CONTEXT_CHAR_LIMIT] + "…"
        lines.append(f"@{row.get('username')}: {content}")
    context_block = "\n".join(lines)
    return (
        f"Recent chat context:\n{context_block}\n\n"
        f"Message to classify (only this):\n«{text}»"
    )


class OverheardNode:
    """Classify an overheard bot-word mention and confirm an insult before acting on it."""

    def __init__(self, llm: Runnable | None = None, confirm_llm: Runnable | None = None) -> None:
        """Build the cheap and confirmation LLMs unless they are injected.

        Args:
            llm: Cheap classifier; built from ``config.FILTER_MODEL`` when None.
            confirm_llm: Insult confirmation LLM; built from
                ``config.INSULT_CONFIRM_MODEL`` when None.
        """
        self.__llm = llm if llm is not None else make_filter_llm(config.FILTER_MODEL)
        self.__confirm_llm = (
            confirm_llm if confirm_llm is not None else make_filter_llm(config.INSULT_CONFIRM_MODEL)
        )

    async def __call__(self, state: BotState) -> dict:
        """Classify the overheard message.

        Args:
            state: Current pipeline state with ``response_trigger="insult_check"``.

        Returns:
            ``{"filter_verdict": Verdict.BOT_INSULT}`` for a confirmed insult;
            otherwise a drop update with ``should_respond=False``.
        """
        msg = state["incoming"]
        text = msg["raw_text"] or ""
        recent = await self.__fetch_recent_context(msg)
        overheard_input = build_overheard_input(text, recent)
        decision = await ask_label(self.__llm, overheard_input, OVERHEARD_SYSTEM)
        if decision == Verdict.BOT_INSULT and await self.__confirm_insult(overheard_input):
            return {"filter_verdict": Verdict.BOT_INSULT}
        extract_passive_facts(msg)
        return {"should_respond": False, "drop_reason": "overheard_dropped"}

    async def __fetch_recent_context(self, msg: dict) -> list[dict]:
        """Load recent chat rows for the overheard classifier, failing soft.

        Args:
            msg: IncomingMessage dict of the message being classified.

        Returns:
            Recent rows (newest-first, current message excluded), or an empty
            list when the lookup fails — degrading to context-free
            classification.
        """
        try:
            recent = await unified_messages.get_recent(
                chat_id=msg["chat_id"], limit=OVERHEARD_CONTEXT_LIMIT + 1
            )
            return [row for row in recent if row["message_id"] != msg["message_id"]]
        except Exception as err:  # noqa: BLE001 — fail soft to context-free classification
            logger.warning("Filter[overheard]: failed to load overheard context: %s", err)
            return []

    async def __confirm_insult(self, overheard_input: str) -> bool:
        """Second opinion from the stronger model before an overheard comeback.

        Fail-soft is inverted here: for an unaddressed aggressive action the
        safe failure is silence, so any confirmation error drops the insult.

        Args:
            overheard_input: The exact input the cheap classifier saw.

        Returns:
            True only when the confirmation model also says BOT_INSULT.
        """
        try:
            response = await self.__confirm_llm.ainvoke([
                SystemMessage(content=OVERHEARD_SYSTEM),
                HumanMessage(content=overheard_input),
            ])
            verdict = response.content.strip().upper()
            logger.debug("Filter[overheard]: overheard insult confirmation verdict: %s", verdict)
            return "INSULT" in verdict
        except Exception as err:  # noqa: BLE001 — any confirmation error resolves to silence
            logger.warning("Insult confirmation failed — dropping overheard insult: %s", err)
            return False

