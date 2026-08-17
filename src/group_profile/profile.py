"""Group-profile orchestration: roster -> LLM verdicts -> rendered message.

One rubric, applied to every known member in one LLM call — the same shape
whether the rubric is "assign X-Men roles" or "score happiness 1-10". See
docs/superpowers/specs/2026-08-17-group-profiling-design.md.
"""

from src.group_profile.generate import fill_missing_verdicts, generate_verdicts
from src.group_profile.render import build_message
from src.group_profile.roster import gather_roster


async def run_group_profile(chat_id: int, rubric: str) -> str | None:
    """Produce the rendered group-profile message for one chat.

    Args:
        chat_id: Chat to profile.
        rubric: The user's own request text, applied verbatim as the theme.

    Returns:
        The full message text ready to send, or None when the chat has no
        members at all to profile.
    """
    roster = await gather_roster(chat_id)
    if not roster.materials_by_uid and not roster.unknown_uids:
        return None
    verdicts = await generate_verdicts(rubric, roster.materials_by_uid)
    verdicts = await fill_missing_verdicts(
        list(roster.materials_by_uid), rubric, roster.materials_by_uid, verdicts
    )
    text = build_message(verdicts, roster.unknown_uids, roster.names_by_uid)
    return text or None
