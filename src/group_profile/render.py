"""Deterministic message assembly for group-profile results.

Members with no material never reach the LLM (see roster.gather_roster) —
their lines come from GROUP_PROFILE_UNKNOWN_LINES instead, so an invented
profile for a ghost member is impossible by construction.
"""

import random

from src.config.prompts import GROUP_PROFILE_UNKNOWN_LINES


def render_verdict_block(name, entry: dict) -> str:
    """Render one member's verdict block: headline plus the reason as a one-liner.

    Args:
        name: Display name (or user_id fallback) shown after the ``@``.
        entry: The member's ``{"verdict", "reason"}`` mapping.

    Returns:
        A two-line block (``@name — verdict`` then the reason), or just the
        headline when no reason is available.
    """
    verdict = entry["verdict"]
    reason = entry.get("reason") or ""
    if reason:
        return f"@{name} — {verdict}\n{reason}"
    return f"@{name} — {verdict}"


def render_unknown_block(name) -> str:
    """Render the canned honest line for a member with no dossier."""
    return f"@{name} — {random.choice(GROUP_PROFILE_UNKNOWN_LINES)}"


def build_message(
    verdicts: dict[int, dict],
    unknown_uids: list[int],
    names_by_uid: dict[int, str],
) -> str:
    """Assemble the full group-profile message from decided verdicts and unknowns.

    Args:
        verdicts: user_id to ``{"verdict", "reason"}`` for members the LLM covered.
        unknown_uids: Members with no material, rendered from the canned pool.
        names_by_uid: Display name for every member appearing in either input.

    Returns:
        The full message text, or an empty string when there is nothing to show.
    """
    blocks = [
        render_verdict_block(names_by_uid.get(user_id, user_id), entry)
        for user_id, entry in verdicts.items()
    ]
    blocks += [
        render_unknown_block(names_by_uid.get(user_id, user_id))
        for user_id in unknown_uids
    ]
    if not blocks:
        return ""
    return "\n\n".join(blocks)
