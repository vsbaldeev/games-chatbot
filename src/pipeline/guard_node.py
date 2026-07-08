"""
GuardNode — prompt-injection gate that runs before the main LLM.

Output contract (verified against the live API on 2026-07-04): the guard
model returns a **numeric probability-of-malicious** string — an injection
attempt scored ``0.9996``, a benign Russian greeting ``0.0004`` — not the
``MALICIOUS``/``BENIGN`` labels the original matcher expected. The node
parses the score and blocks at ``GUARD_SCORE_THRESHOLD``.

Scope: only text the user actually typed is classified — ``raw_text``, which
carries the message text or the media caption. Transcripts and vision
descriptions are produced by our own models and scanning them yields only
false positives; the response prompt's identity rules are the defence layer
for spoken imperatives.

On a malicious verdict with an explicit trigger the node answers with a
neutral deflection (no guilt presumed — gaming-chat idiom is structurally
close to jailbreak phrasing, so false positives are inherent at 86M
parameters). The permanent hack-attempt memory fact is recorded only on the
**second** flag per (chat, user) within 24 hours: blocking is reversible and
cheap, the memory fact is neither.

Fails open: if the guard API is unavailable or returns an unparsable label,
the message is allowed through rather than blocking all chat traffic.
"""

import asyncio
import random

from groq import AsyncGroq

from src import config, log
from src.pipeline.state import BotState
from src.store import user_memories
from src.utils.ttl_gate import TtlGate

logger = log.get_logger(__name__)

# Probability-of-malicious above which the message is blocked.
GUARD_SCORE_THRESHOLD = 0.9

# The permanent memory fact fires only on a repeat flag within this window.
GUARD_FLAG_WINDOW_SECONDS = 24 * 60 * 60
HACK_FLAGS_BEFORE_FACT = 2
guard_flag_gate = TtlGate(GUARD_FLAG_WINDOW_SECONDS)

# Neutral deflections that work for a real injection attempt and an innocent
# false positive alike — in character, no guilt presumed.
GUARD_REFUSALS = [
    "Не, это без меня.",
    "Так, это мимо меня.",
    "В это я не играю.",
    "Давай что-нибудь попроще.",
    "Пас. Следующий вопрос.",
]


class GuardNode:
    """Runs prompt-injection classification and short-circuits blocked messages."""

    async def __call__(self, state: BotState) -> dict:
        """Classify the typed text and block the pipeline on a malicious score.

        Args:
            state: Current pipeline state.

        Returns:
            State update dict with ``blocked`` and, for blocked explicit
            triggers, a canned deflection in ``response``.
        """
        msg = state["incoming"]
        # Typed text only: for media messages raw_text carries the caption;
        # a media message without a caption skips classification entirely.
        text = msg.get("raw_text") or ""

        if not text.strip():
            return {"blocked": False}

        if await self.__classify(text):
            logger.warning(
                "Guard blocked message from @%s in chat %s (trigger: %s)",
                msg["username"], msg["chat_id"], state.get("response_trigger"),
            )
            if state.get("response_trigger") == "explicit":
                flags = guard_flag_gate.hit((msg["chat_id"], msg["user_id"]))
                if flags >= HACK_FLAGS_BEFORE_FACT:
                    asyncio.create_task(self.__record_hack_attempt(msg))
                return {"blocked": True, "response": random.choice(GUARD_REFUSALS)}
            return {"blocked": True}

        return {"blocked": False}

    async def __classify(self, text: str) -> bool:
        """Score the text with the guard model.

        Args:
            text: Typed message text or media caption.

        Returns:
            True when the returned malicious-probability score reaches
            ``GUARD_SCORE_THRESHOLD``. Fails open to False on API errors or
            an unparsable label.
        """
        try:
            client = AsyncGroq(api_key=config.GROQ_API_KEY, max_retries=0)
            result = await client.chat.completions.create(
                model=config.GUARD_MODEL,
                messages=[{"role": "user", "content": text}],
            )
            raw_label = result.choices[0].message.content.strip()
            logger.info("Guard raw label: %s", raw_label)
            return float(raw_label) >= GUARD_SCORE_THRESHOLD
        except Exception as err:
            logger.warning("Guard classification failed, failing open: %s", err)
            return False

    async def __record_hack_attempt(self, msg) -> None:
        """Increment the hack-attempt counter fact for the flagged user.

        Args:
            msg: IncomingMessage dict of the blocked message.
        """
        try:
            await user_memories.upsert_hack_attempt(
                chat_id=msg["chat_id"],
                user_id=msg["user_id"],
                username=msg["username"],
            )
        except Exception as err:
            logger.warning("Failed to record hack attempt for @%s: %s", msg["username"], err)
