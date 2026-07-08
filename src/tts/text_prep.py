"""Reply-text preparation for speech synthesis.

Decides whether a reply is speakable at all and normalizes it for Silero:
URLs become the spoken word «ссылка», emojis and pictographs are dropped.
"""

import re

from src import config

CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)

TTS_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Superset of the reply-emoji ranges in events/messages.py: adds arrows,
# dingbats and misc symbols (U+2190–U+2BFF), variation selectors and the
# zero-width joiner that glue composite emojis together.
TTS_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002190-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]"
)


def prepare_tts_text(reply_text: str) -> str | None:
    """Normalize a reply for synthesis or reject it as unspeakable.

    Args:
        reply_text: Markdown-stripped, homoglyph-normalized reply text.

    Returns:
        Synthesis-ready text, or None when the reply should stay text:
        empty after cleaning, no Cyrillic letters (Silero v5_ru raises on
        pure Latin/digit input), or longer than ``config.TTS_MAX_CHARS``.
    """
    cleaned = TTS_URL_RE.sub("ссылка", reply_text)
    cleaned = TTS_EMOJI_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if not CYRILLIC_RE.search(cleaned):
        return None
    if len(cleaned) > config.TTS_MAX_CHARS:
        return None
    return cleaned
