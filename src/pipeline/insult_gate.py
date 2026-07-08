"""Per-user escalation ladder for insults aimed at the bot.

Decides how much attention a confirmed insult deserves, so the bot entertains
the chat without feeding a flame war ("don't feed the troll"). State is
intentionally in-memory per (chat_id, user_id) — mirroring ``humor_gate`` —
because losing the ladder position on restart is harmless.

Ladder within a rolling window:
  1st insult        → COMEBACK_TIER   (full witty comeback via the response LLM)
  2nd–3rd insult    → DISMISSIVE_TIER (canned bored one-liner, no LLM)
  4th and beyond    → IGNORE_TIER     (dismissive emoji reaction, nothing more)
"""

import time

WINDOW_SECONDS = 30 * 60   # ladder position decays after 30 minutes of peace
DISMISSIVE_MAX_COUNT = 3   # insults 2..3 in the window get a canned one-liner

COMEBACK_TIER = 1
DISMISSIVE_TIER = 2
IGNORE_TIER = 3

insult_times: dict[tuple[int, int], list[float]] = {}


def register_insult(chat_id: int, user_id: int) -> int:
    """Record one confirmed insult and return the response tier for it.

    Prunes timestamps older than ``WINDOW_SECONDS``, appends the current one,
    and maps the resulting in-window count onto the ladder.

    Args:
        chat_id: Chat the insult was posted in.
        user_id: Author of the insult.

    Returns:
        ``COMEBACK_TIER`` for the first insult in the window,
        ``DISMISSIVE_TIER`` for insults 2..``DISMISSIVE_MAX_COUNT``,
        ``IGNORE_TIER`` beyond that.
    """
    now = time.time()
    key = (chat_id, user_id)
    recent = [moment for moment in insult_times.get(key, []) if now - moment < WINDOW_SECONDS]
    recent.append(now)
    insult_times[key] = recent

    count = len(recent)
    if count == 1:
        return COMEBACK_TIER
    if count <= DISMISSIVE_MAX_COUNT:
        return DISMISSIVE_TIER
    return IGNORE_TIER
