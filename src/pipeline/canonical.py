"""Canonical log line: one INFO summary per message handled by the pipeline.

Implements the "canonical log line" pattern — instead of stitching together
per-node log chatter, a single key=value line at the end of the pipeline run
answers "what came in and what did the bot do about it".
"""

from src import log
from src.pipeline.state import BotState

logger = log.get_logger("pipeline")

MISSING = "-"


def derive_drop_reason(state: BotState) -> str | None:
    """Work out why the pipeline produced no reply, when nodes did not say.

    Args:
        state: Final pipeline state.

    Returns:
        An explicit ``drop_reason`` set by a node, ``"not_addressed"`` when
        the router decided not to respond, or None when unknown.
    """
    explicit = state.get("drop_reason")
    if explicit:
        return explicit
    if not state.get("should_respond") and not state.get("filter_verdict"):
        return "not_addressed"
    return None


def emit(state: BotState, action: str, elapsed_seconds: float) -> None:
    """Log the canonical one-line summary for a pipeline run.

    Args:
        state: Final (or best-available) pipeline state.
        action: Outcome tag: ``replied``, ``joked``, ``ignored`` or
            ``error:<kind>``.
        elapsed_seconds: Wall-clock duration of the pipeline run.
    """
    incoming = state["incoming"]
    guard = MISSING
    if state.get("blocked"):
        guard = "block"
    elif state.get("filter_verdict"):
        guard = "ok"
    fields = [
        f"chat={incoming['chat_id']}",
        f"user=@{incoming['username']}",
        f"kind={incoming['media_type']}",
        f"msg={incoming['message_id']}",
        f"trigger={state.get('response_trigger') or MISSING}",
        f"filter={state.get('filter_verdict') or MISSING}",
        f"tier={state.get('engagement_tier', MISSING)}",
        f"guard={guard}",
        f"action={action}",
    ]
    reason = derive_drop_reason(state)
    if reason and action == "ignored":
        fields.append(f"reason={reason}")
    response = state.get("response") or ""
    if response.strip():
        fields.append(f"len={len(response)}")
    fields.append(f"dur={elapsed_seconds:.2f}s")
    logger.info(" ".join(fields))
