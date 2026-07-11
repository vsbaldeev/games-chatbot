"""Silent daily current-activity refresh.

Runs once a day (see ``src/jobs/daily_activity.py``) and invents a new
present-tense activity phrase for Жора without posting anything to chat —
purely so ``bot_memories.get_current_activity()`` has a fresh answer for
«что делаешь?» every day instead of only on the 2x/week life-post schedule.
Mirrors ``src/life/poster.py:distill_facts`` for the direct-``ChatGroq`` +
``ainvoke_with_backoff`` call pattern.
"""

import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent.middleware import ainvoke_with_backoff, strip_thinking
from src.config.prompts import DAILY_ACTIVITY_SYSTEM
from src.life import calendar_ru
from src.life.writer import CURRENT_ACTIVITY_MAX_CHARS
from src.store import bot_memories

logger = log.get_logger(__name__)

RECENT_ACTIVITY_CONTEXT = 10
ACTIVITY_FACTS_CONTEXT = 5
ACTIVITY_WRITE_ATTEMPTS = 2


def build_activity_prompt(
    recent_activities: list[tuple[str, float]], facts: list[str], now: datetime.datetime
) -> str:
    """Assemble the human turn for the daily activity refresh.

    Args:
        recent_activities: Recent ``(phrase, posted_at)`` pairs, newest
            first, as returned by ``bot_memories.get_recent_activities``.
        facts: Canon facts to ground the new activity (newest first).
        now: Current moment, already in Moscow Time.

    Returns:
        The prompt string to send as the human turn.
    """
    parts = [f"Сегодня {calendar_ru.describe_moscow_date(now)} (по Москве).", ""]
    if recent_activities:
        parts.append("Твои недавние занятия (от новых к старым, не повторяй их):")
        parts.extend(
            f"- {calendar_ru.describe_relative_day(posted_at, now)} — {phrase}"
            for phrase, posted_at in recent_activities
        )
        parts.append("")
    if facts:
        parts.append("Факты твоего канона:")
        parts.extend(f"- {fact}" for fact in facts)
        parts.append("")
    parts.append("Придумай новое занятие на сегодня. Ответь одной фразой.")
    return "\n".join(parts)


def parse_activity_phrase(raw: str) -> str | None:
    """Parse and validate the model's raw output into a usable activity phrase.

    Args:
        raw: Raw model output, possibly wrapped in a think block.

    Returns:
        A trimmed phrase within the character limit, or None when the
        output is empty or too long — a retry should follow, not a
        silent truncation.
    """
    cleaned = strip_thinking(raw)
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    phrase = first_line.strip("«»\"' .")
    if not phrase or len(phrase) > CURRENT_ACTIVITY_MAX_CHARS:
        return None
    return phrase


def make_activity_llm() -> ChatGroq:
    """Build the ChatGroq instance used for the daily activity refresh.

    Returns:
        Configured ChatGroq client — high temperature for variety, no
        retries (``ainvoke_with_backoff`` handles transient rate limits).
    """
    return ChatGroq(
        model=config.ACTIVITY_MODEL, api_key=config.GROQ_API_KEY,
        temperature=0.9, max_tokens=100, max_retries=0,
    )


async def refresh_daily_activity() -> None:
    """Generate and persist today's activity phrase.

    Fails soft: if every attempt produces an unusable response, the
    previous activity is left in place and simply ages into "recent"
    phrasing — a broken refresh must never crash the scheduled job.
    """
    try:
        recent_activities = await bot_memories.get_recent_activities(RECENT_ACTIVITY_CONTEXT)
        facts = await bot_memories.get_facts(ACTIVITY_FACTS_CONTEXT)
        now = datetime.datetime.now(calendar_ru.MOSCOW_TZ)
        prompt = build_activity_prompt(recent_activities, facts, now)
        llm = make_activity_llm()
        for attempt in range(ACTIVITY_WRITE_ATTEMPTS):
            result = await ainvoke_with_backoff(
                llm, [SystemMessage(content=DAILY_ACTIVITY_SYSTEM), HumanMessage(content=prompt)],
            )
            phrase = parse_activity_phrase(result.content or "")
            if phrase is not None:
                await bot_memories.insert_activity(phrase)
                return
            logger.warning("Daily activity refresh produced unusable output (attempt %d)", attempt + 1)
    except Exception as error:
        logger.warning("Daily activity refresh failed: %s", error)
