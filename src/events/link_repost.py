"""One-message link repost: video + summary caption, then delete the original.

A link message used to produce three messages in chat — the user's link, the
bot's bare video repost, and the bot's separate text reply. This module
collapses the bot's two into one and, when the user's message was nothing but
the link, removes that too.

The delete only ever happens after a successful send. A send that returns a
``telegram.Message`` is the delivery confirmation; there is nothing further to
check. If the send fails the original message stays and the summary still goes
out as a plain text reply — the download and LLM spend already happened.
"""

import re

from src import log
from src.agent.compress import compress_to_budget

logger = log.get_logger(__name__)

CAPTION_LIMIT = 1024  # Telegram Bot API cap on media captions

SENTENCE_END_RE = re.compile(r"[.!?…]")


def build_caption(summary: str, username: str | None, url: str | None) -> str:
    """Compose the caption for the bot's single message.

    Args:
        summary: The pipeline's summary text.
        username: Sender to credit, or None when the original message is
            surviving and already shows who posted it.
        url: Canonical link to carry, or None for the same reason.

    Returns:
        Credit line + link + summary when the original is being deleted, the
        bare summary otherwise.
    """
    if username is None or url is None:
        if username is not None or url is not None:
            logger.warning(
                "Link caption got a half-filled credit pair (username=%s, url=%s); "
                "falling back to the bare summary",
                username, url,
            )
        return summary
    return f"Скинул @{username}\n{url}\n\n{summary}"


def truncate_at_sentence(text: str, budget: int) -> str:
    """Cut ``text`` to ``budget`` characters at the last sentence boundary.

    The unreachable-in-practice backstop for
    :func:`fit_caption`: it runs only when the compressor already failed to
    hit the budget, and exists so delivery can never raise.

    Args:
        text: Text to shorten.
        budget: Hard character ceiling.

    Returns:
        Text of at most ``budget`` characters, ending at a sentence boundary
        when one exists inside the budget and with an ellipsis when not.
    """
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    head = text[:budget]
    boundaries = [match.end() for match in SENTENCE_END_RE.finditer(head)]
    if boundaries:
        return head[:boundaries[-1]].strip()
    return head[:budget - 1].rstrip() + "…"


async def fit_caption(summary: str, username: str | None, url: str | None) -> str:
    """Compose a caption guaranteed to fit Telegram's caption limit.

    Three rungs: send as composed when it already fits; otherwise compress the
    summary against the budget left by the credit line and link; and only if
    the compressor still overshoots, truncate at a sentence boundary.

    Args:
        summary: The pipeline's summary text.
        username: Sender to credit, or None — see :func:`build_caption`.
        url: Canonical link to carry, or None — see :func:`build_caption`.

    Returns:
        A caption of at most :data:`CAPTION_LIMIT` characters.
    """
    caption = build_caption(summary, username, url)
    if len(caption) <= CAPTION_LIMIT:
        return caption
    budget = CAPTION_LIMIT - (len(caption) - len(summary))
    if budget <= 0:
        # The credit line and URL alone overflow the cap. Canonical link URLs
        # run about 50 characters, so this is unreachable in practice — but
        # the return contract is absolute and the Bot API rejects anything
        # longer, so hand back something it will accept.
        logger.warning("Link caption overhead alone exceeds the caption limit")
        return caption[:CAPTION_LIMIT]
    compressed = await compress_to_budget(summary, budget)
    caption = build_caption(compressed, username, url)
    if len(caption) <= CAPTION_LIMIT:
        return caption
    logger.warning("Caption still over the limit after compression, truncating")
    return build_caption(truncate_at_sentence(compressed, budget), username, url)
