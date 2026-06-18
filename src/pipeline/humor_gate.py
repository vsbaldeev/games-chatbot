"""Cheap, no-LLM opportunity gate for autonomous humor.

Keeps the expensive comedian model off the per-message hot path. Per-chat
in-memory counters decide whether a message is even worth *considering* for a
joke; the model only runs when the gate fires. State is intentionally in-memory
(KISS, mirroring ``offense_reply_counts``) — losing cadence history on restart is
harmless for a rare-and-sharp feature.

Cadence (anti-spam):
  - at least ``MIN_MESSAGES_SINCE_JOKE`` messages must pass since the last
    consideration, so the bot opens lulls rather than talking over a live thread;
  - the ``COOLDOWN_SECONDS`` window must have elapsed since the last *sent* joke,
    so the bot never piles its own jokes on top of one another;
  - even when eligible, a low probability roll keeps it rare.
"""

import random
import time

from src.pipeline.state import IncomingMessage

MIN_MESSAGE_LEN = 30            # message must carry some substance to be joke-worthy
MIN_MESSAGES_SINCE_JOKE = 15    # let the chat breathe between jokes
COOLDOWN_SECONDS = 120 * 60      # at most ~one joke per 2 hours per chat
CONSIDER_PROBABILITY = 0.15     # even when eligible, usually stay quiet

messages_since_joke: dict[int, int] = {}
last_joke_time: dict[int, float] = {}


def observe(chat_id: int) -> None:
    """Count one incoming chat message toward the next-joke gap.

    Args:
        chat_id: Chat the message belongs to.
    """
    messages_since_joke[chat_id] = messages_since_joke.get(chat_id, 0) + 1


def should_consider(chat_id: int, msg: IncomingMessage) -> bool:
    """Return True when this moment is worth handing to the comedian.

    Pure predicate — never mutates state. All of the following must hold: plain
    joke-worthy text, enough messages since the last joke, the cooldown elapsed,
    and a low probability roll.

    Args:
        chat_id: Chat the message belongs to.
        msg: The incoming message.

    Returns:
        True to route the message to the humor node.
    """
    if msg["media_type"] != "text" or msg.get("is_forwarded"):
        return False
    text = (msg.get("raw_text") or "").strip()
    if len(text) < MIN_MESSAGE_LEN:
        return False
    if messages_since_joke.get(chat_id, 0) < MIN_MESSAGES_SINCE_JOKE:
        return False
    if time.time() - last_joke_time.get(chat_id, 0.0) < COOLDOWN_SECONDS:
        return False
    return random.random() < CONSIDER_PROBABILITY


def mark_considered(chat_id: int) -> None:
    """Reset the message gap after the comedian was consulted (joke or not).

    Prevents re-invoking the model on the next message; the chat must accumulate
    another ``MIN_MESSAGES_SINCE_JOKE`` messages before the gate can fire again.

    Args:
        chat_id: Chat the consideration happened in.
    """
    messages_since_joke[chat_id] = 0


def mark_joke_sent(chat_id: int) -> None:
    """Reset the gap and start the cooldown after a joke is actually sent.

    The bot then stays quiet and lets the chat run with the joke.

    Args:
        chat_id: Chat the joke was sent to.
    """
    messages_since_joke[chat_id] = 0
    last_joke_time[chat_id] = time.time()
