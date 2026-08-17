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

import io
import re

from src import log
from src.agent.compress import compress_to_budget

logger = log.get_logger(__name__)

CAPTION_LIMIT = 1024  # Telegram Bot API cap on media captions
TEXT_LIMIT = 4096  # Telegram Bot API cap on a plain text message

SENTENCE_END_RE = re.compile(r"[.!?…]")


def build_cap_remaining_footer(cap_remaining: int | None) -> str | None:
    """Render the daily-cap remaining-count line, or None to omit it.

    Args:
        cap_remaining: Reels left in today's cap after this one, or None
            when the matched handler has no cap (see
            ``social_links.LinkHandler.daily_cap``) — only Instagram's
            handler has one today.

    Returns:
        A one-line status string, or None when there is nothing to show.
    """
    if cap_remaining is None:
        return None
    return f"Осталось рилсов сегодня: {cap_remaining}"


def build_caption(
    summary: str, username: str | None, url: str | None, footer: str | None = None,
) -> str:
    """Compose the caption for the bot's single message.

    Args:
        summary: The pipeline's summary text.
        username: Sender to credit, or None when the original message is
            surviving and already shows who posted it.
        url: Canonical link to carry, or None for the same reason.
        footer: Fixed trailer appended after the summary (e.g. the daily-cap
            remaining-count line) — never compressed or truncated away, same
            as the credit line.

    Returns:
        Credit line + link + summary when the original is being deleted, the
        bare summary otherwise, plus ``footer`` when given.
    """
    if username is None or url is None:
        if username is not None or url is not None:
            logger.warning(
                "Link caption got a half-filled credit pair (username=%s, url=%s); "
                "falling back to the bare summary",
                username, url,
            )
        body = summary
    else:
        body = f"Скинул @{username}\n{url}\n\n{summary}"
    return f"{body}\n\n{footer}" if footer else body


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


async def fit_caption(
    summary: str, username: str | None, url: str | None, has_video: bool,
    footer: str | None = None,
) -> str:
    """Compose a caption guaranteed to fit Telegram's relevant length limit.

    Three rungs: send as composed when it already fits; otherwise compress the
    summary against the budget left by the credit line, link and footer; and
    only if the compressor still overshoots, truncate at a sentence boundary.

    Args:
        summary: The pipeline's summary text.
        username: Sender to credit, or None — see :func:`build_caption`.
        url: Canonical link to carry, or None — see :func:`build_caption`.
        has_video: True when the message is sent as a video caption (capped
            at :data:`CAPTION_LIMIT`); False when it is a plain text message
            (capped at the much larger :data:`TEXT_LIMIT`).
        footer: Fixed trailer — see :func:`build_caption`.

    Returns:
        A caption of at most :data:`CAPTION_LIMIT` characters when
        ``has_video`` is True, or :data:`TEXT_LIMIT` characters otherwise.
    """
    limit = CAPTION_LIMIT if has_video else TEXT_LIMIT
    caption = build_caption(summary, username, url, footer)
    if len(caption) <= limit:
        return caption
    budget = limit - (len(caption) - len(summary))
    if budget <= 0:
        # The credit line, URL and footer alone overflow the cap. Canonical
        # link URLs run about 50 characters and the footer about 30, so this
        # is unreachable in practice — but the return contract is absolute
        # and the Bot API rejects anything longer, so hand back something it
        # will accept.
        logger.warning("Link caption overhead alone exceeds the caption limit")
        return caption[:limit]
    compressed = await compress_to_budget(summary, budget)
    caption = build_caption(compressed, username, url, footer)
    if len(caption) <= limit:
        return caption
    logger.warning("Caption still over the limit after compression, truncating")
    return build_caption(truncate_at_sentence(compressed, budget), username, url, footer)


async def send_combined(msg, caption: str, video: bytes | None, anchored: bool):
    """Send the one combined message, as a reply or on its own.

    The un-anchored branches post directly on ``msg.chat`` rather than
    replying, so they need ``message_thread_id`` spelled out explicitly to
    land in the right forum topic — ``reply_video``/``reply_text`` get that
    for free from python-telegram-bot because they reply to ``msg`` itself.

    Args:
        msg: The triggering ``telegram.Message``.
        caption: Caption for the video, or the whole body when there is none.
        video: Downloaded video bytes, or None for a text-only link.
        anchored: True to reply to ``msg``, False to post un-anchored because
            ``msg`` is about to be deleted.

    Returns:
        Tuple of the sent ``telegram.Message`` and its media type.

    Raises:
        Exception: Whatever the Bot API raises on a failed send; the caller
            turns that into the text fallback.
    """
    thread_id = msg.message_thread_id if msg.is_topic_message else None
    if video:
        payload = io.BytesIO(video)
        if anchored:
            return await msg.reply_video(video=payload, caption=caption), "video"
        return await msg.chat.send_video(
            video=payload, caption=caption, message_thread_id=thread_id,
        ), "video"
    if anchored:
        return await msg.reply_text(caption), "text"
    return await msg.chat.send_message(caption, message_thread_id=thread_id), "text"


async def try_delete_original(msg) -> None:
    """Delete the user's link message, tolerating a missing permission.

    Deleting another member's message needs the bot to be an admin with
    ``can_delete_messages``. Without it the chat is merely left with a
    redundant link — the summary is already delivered, so there is nothing to
    roll back.

    Args:
        msg: The triggering ``telegram.Message`` to remove.
    """
    try:
        await msg.delete()
    except Exception as error:
        logger.warning("Could not delete the original link message: %s", error)


def resolve_bare_deletion(is_bare: bool, url: str | None) -> bool:
    """Downgrade a bare deletion to a kept original when the URL is missing.

    A missing canonical URL means deleting the trigger message would strand
    the link nowhere in the chat, so a bare delete is only allowed when both
    conditions hold.

    Args:
        is_bare: Whether the trigger message was the link and nothing else.
        url: Canonical link resolved for this trigger, or None.

    Returns:
        ``is_bare`` unchanged, unless it was True with no URL, in which case
        False.
    """
    if is_bare and url is None:
        logger.warning(
            "Bare link message has no canonical URL; keeping the original "
            "instead of deleting it and losing the link"
        )
        return False
    return is_bare


async def deliver_link_message(
    msg, *, summary: str, video: bytes | None, username: str,
    url: str | None, is_bare: bool, cap_remaining: int | None = None,
) -> tuple[int, int | None, str]:
    """Deliver a link summary as one message, deleting the original if bare.

    Args:
        msg: The triggering ``telegram.Message``.
        summary: Markdown-stripped summary produced by the pipeline.
        video: Downloaded video bytes, or None (long-form YouTube, or a failed download).
        username: Sender's display name, credited only when deleting.
        url: Canonical link, carried only when deleting; see :func:`resolve_bare_deletion`.
        is_bare: True when the message was the link and nothing else.
        cap_remaining: Reels left in today's cap, or None — see
            :func:`build_cap_remaining_footer`. Shown regardless of
            ``is_bare``, unlike the credit line.

    Returns:
        On success: sent message id, anchor id (None when un-anchored), and media
        type. On a send failure that falls back to text: fallback id,
        ``msg.message_id``, and ``"text"`` — always, since no delete follows one.

    Raises:
        Exception: Re-raised when the failed send was already the anchored text reply.
    """
    is_bare = resolve_bare_deletion(is_bare, url)
    caption = await fit_caption(
        summary, username if is_bare else None, url if is_bare else None,
        has_video=bool(video), footer=build_cap_remaining_footer(cap_remaining),
    )
    try:
        sent, media_type = await send_combined(msg, caption, video, anchored=not is_bare)
    except Exception as error:
        if video is None and not is_bare:
            # This failed call was already the anchored text reply; retrying
            # it would just raise again. Let the caller handle it.
            raise
        logger.warning("Combined link message failed, replying with text: %s", error)
        fallback = await msg.reply_text(summary)
        return fallback.message_id, msg.message_id, "text"
    if not is_bare:
        return sent.message_id, msg.message_id, media_type
    await try_delete_original(msg)
    return sent.message_id, None, media_type


def resolve_link_delivery(final_state) -> tuple[bytes | None, str | None] | None:
    """Pull the link-repost inputs out of the finished pipeline state.

    Returns None unless the trigger was a link *and* the fetch produced a
    content block — a failed download or fetch leaves the user's message
    alone rather than trading it for an empty summary.

    Args:
        final_state: Pipeline state after the graph run.

    Returns:
        ``(video_bytes, canonical_url)`` for a deliverable link, else None.
    """
    trigger = final_state.get("response_trigger")
    if trigger == "youtube_short" and final_state.get("youtube_short_content"):
        return final_state.get("youtube_short_video"), final_state.get("youtube_short_url")
    if trigger == "social_link" and final_state.get("social_link_content"):
        return final_state.get("social_link_video"), final_state.get("social_link_url")
    return None
