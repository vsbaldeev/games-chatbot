"""Voice-note replies — Telegram-facing glue over the TTS service.

Keeps ``src.tts`` free of Telegram knowledge: this module owns the chat
action, the ``reply_voice`` call and the never-raise contract that lets
callers fall back to a plain text reply.
"""

import io

from src import log
from src.tts import prepare_tts_text, speech_service

logger = log.get_logger(__name__)


async def try_send_voice_reply(msg, reply_text: str):
    """Attempt to answer a voice-triggered message with a Telegram voice note.

    Args:
        msg: The triggering ``telegram.Message`` to reply to.
        reply_text: Markdown-stripped, homoglyph-normalized reply text.

    Returns:
        The sent ``telegram.Message`` on success, or None on any failure or
        ineligibility (service not ready, text unsuitable for synthesis,
        synthesis or send error) — the caller then sends the text reply.
    """
    if not speech_service.is_ready:
        return None
    prepared_text = prepare_tts_text(reply_text)
    if prepared_text is None:
        return None
    try:
        await msg.chat.send_action("record_voice")
        voice = await speech_service.synthesize(prepared_text)
        if voice is None:
            return None
        return await msg.reply_voice(
            voice=io.BytesIO(voice.ogg_bytes), duration=voice.duration_seconds
        )
    except Exception as error:
        logger.warning("Voice reply failed, falling back to text: %s", error)
        return None
