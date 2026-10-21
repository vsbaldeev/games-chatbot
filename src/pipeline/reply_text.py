"""Cleanup of the model's reply text before it is delivered: markdown, leaked speaker labels, writing tics."""

import re


TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


# Matches a leaked history speaker label at the start of a reply — see
# strip_speaker_prefix. Covers both "@username:" and "Ты (бот):", plus the
# reply-chain arrow build_trigger_line appends to the sender's own label.
SPEAKER_PREFIX_RE = re.compile(r"^\s*(?:Ты\s*\(бот\)|@[A-Za-z0-9_]+)(?:\s*\(↳.*\))?\s*:\s*")


# Scare quotes around a single word — see strip_writing_tics.
SCARE_QUOTE_RE = re.compile(r"«(\S{1,24})»")


DASH_MAP = str.maketrans({"‑": "-", "–": "-"})


TRAILING_EMOJI_RE = re.compile(r"[\s‍️☀-➿\U0001F300-\U0001FAFF]+$")


def strip_markdown(text: str) -> str:
    """Strip common Markdown formatting characters from text.

    Removes bold, italic, and table-separator lines so the output reads as
    plain chat text rather than formatted markup.

    Args:
        text: Input string that may contain Markdown.

    Returns:
        Plain text with bold/italic markers removed and table separators dropped.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text, flags=re.DOTALL)
    lines = [line for line in text.splitlines() if not TABLE_SEP_RE.match(line)]
    return "\n".join(lines)


def strip_speaker_prefix(text: str) -> str:
    """Remove a leaked history speaker label from the start of a reply.

    Recent history and thread history are rendered to the model as
    ``@username: …`` / ``Ты (бот): …`` lines (see :func:`render_row` and
    :func:`build_trigger_line`), and the response model sometimes continues
    that transcript instead of answering — emitting the label as part of its
    own reply (e.g. ``"Ты (бот): Ну да, еблан..."``).

    Args:
        text: Raw response text from the LLM.

    Returns:
        Text with a single leading speaker label removed, if present.
    """
    return SPEAKER_PREFIX_RE.sub("", text, count=1)


def strip_writing_tics(text: str) -> str:
    """Remove register markers no one types in a Telegram chat.

    Guillemets around a single word are scare quotes copied from formal
    written Russian, not chat style — dropped, but multi-word «цитаты» are
    left alone since those are genuine quoting. Non-breaking and en dashes
    are collapsed to a plain hyphen. A trailing run of emoji is dropped
    outright: the persona prompt asks for at most one emoji and only when
    it fits, but the response model appends one to nearly every reply
    regardless.

    Args:
        text: Response text, already stripped of Markdown and speaker prefix.

    Returns:
        Text with scare quotes, atypical dashes, and trailing emoji removed.
    """
    text = SCARE_QUOTE_RE.sub(r"\1", text)
    text = text.translate(DASH_MAP)
    return TRAILING_EMOJI_RE.sub("", text).rstrip()
