"""Opaque user_id <-> anon-key mapping so an LLM never sees real Telegram ids.

Shared by every feature that sends per-member facts to an LLM keyed by
position rather than identity — the weekly-roles job (src/jobs/roles.py) and
group profiling (src/group_profile/generate.py).
"""


def anonymise(user_ids: list[int]) -> tuple[dict[int, str], dict[str, int]]:
    """Map user_ids to opaque ``user_N`` keys so the LLM never sees real ids.

    Args:
        user_ids: User ids to anonymise, in iteration order.

    Returns:
        A ``(uid_to_anon, anon_to_uid)`` pair of inverse mappings.
    """
    uid_to_anon = {user_id: f"user_{index}" for index, user_id in enumerate(user_ids)}
    anon_to_uid = {anon: user_id for user_id, anon in uid_to_anon.items()}
    return uid_to_anon, anon_to_uid
