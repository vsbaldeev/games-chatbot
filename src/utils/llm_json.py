"""Shared helper for parsing a JSON object out of raw LLM output.

Strips code fences models sometimes wrap JSON in, then parses. Any parse
failure is logged with a caller-supplied label so different jobs' log lines
stay distinguishable, and returns None so the caller can skip the slot
instead of crashing.
"""

import json
import re

from src import log

logger = log.get_logger(__name__)

CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)


def load_json_object(raw: str, context: str = "LLM") -> dict | None:
    """Strip code fences and parse the LLM response into a dict, or None on failure.

    Args:
        raw: Raw model response text, possibly wrapped in a markdown code fence.
        context: Short label identifying the caller, used in the warning log
            (e.g. ``"Role generation"``, ``"Episode generation"``).

    Returns:
        The parsed dict, or None when the response is not valid JSON or does
        not parse into a dict.
    """
    try:
        cleaned = CODE_FENCE_RE.sub("", raw.strip())
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, AttributeError) as error:
        logger.warning("%s returned non-JSON: %s — %s", context, error, raw[:200])
        return None
    if not isinstance(data, dict):
        logger.warning("%s expected dict, got %s", context, type(data).__name__)
        return None
    return data
