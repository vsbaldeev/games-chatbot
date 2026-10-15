"""Deriving a delivered message's content kind and its question from the final pipeline state.

Nodes that put a canned text in ``response`` (guard, filter, router) stamp
``content_kind`` themselves. Every other delivered reply was written by the
response LLM, and the directive it obeyed decides the kind.
"""

from src.pipeline.state import BotState, ContentKind

LINK_TRIGGERS = frozenset({"youtube_short", "social_link"})


def content_kind_for(state: BotState) -> ContentKind:
    """Pick the content kind of the message the pipeline is about to deliver.

    Precedence for LLM-written replies: wind-down, then insult, then link
    summary, then a plain response. A wound-down insult carries both flags
    but the wind-down directive is the one the model was told to follow.
    """
    if "content_kind" in state:
        return state["content_kind"]
    if state.get("wind_down"):
        return ContentKind.WIND_DOWN_REPLY
    if state.get("is_bot_insult"):
        return ContentKind.INSULT_REPLY
    if state.get("response_trigger") in LINK_TRIGGERS:
        return ContentKind.LINK_SUMMARY
    return ContentKind.RESPONSE


def question_text(state: BotState) -> str:
    """The member's question as the bot understood it: the transcript, else the typed text, else ''."""
    incoming = state["incoming"]
    return (incoming.get("processed_text") or incoming.get("raw_text") or "").strip()
