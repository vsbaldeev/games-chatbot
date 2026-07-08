"""Language detection and correction utilities."""

import re

from langchain_core.messages import HumanMessage

from src import log
from src.agent.middleware import strip_thinking
from src.config.prompts import LANGUAGE_CORRECTION_PROMPT

logger = log.get_logger(__name__)

FOREIGN_SCRIPT_RE = re.compile(
    "[一-鿿"   # CJK Unified Ideographs
    "㐀-䶿"    # CJK Extension A
    "가-힯"    # Hangul Syllables
    "ᄀ-ᇿ"    # Hangul Jamo
    "぀-ヿ"    # Hiragana + Katakana
    "฀-๿"    # Thai
    "؀-ۿ"    # Arabic
    "֐-׿]"   # Hebrew
)

# Greek block is handled separately from FOREIGN_SCRIPT_RE: unlike CJK/Arabic,
# Greek letters appear legitimately as standalone symbols in Russian text (π,
# 50 μg, alpha/beta versions), so they only warrant correction when spliced into
# a Cyrillic word — the homoglyph-garbling signature.
GREEK_RE = re.compile(
    "[Ͱ-Ͽ"   # Greek and Coptic
    "ἀ-῿]"  # Greek Extended (accented forms)
)

# Confusable letters (Greek/Latin) that the models occasionally emit in place of
# visually identical Cyrillic letters, mapped back to their Cyrillic form. This
# is the subset of the Unicode UTS #39 "confusables" table that is genuinely
# indistinguishable by eye, so applying it inside a mostly-Cyrillic word never
# corrupts legitimate text.
HOMOGLYPH_MAP = {
    # Greek lowercase -> Cyrillic
    "ο": "о",  # ο -> о
    "α": "а",  # α -> а
    "ε": "е",  # ε -> е
    "ρ": "р",  # ρ -> р
    "χ": "х",  # χ -> х
    "κ": "к",  # κ -> к
    # Greek accented vowels -> Cyrillic
    "ά": "а",  # ά -> а
    "έ": "е",  # έ -> е
    "ό": "о",  # ό -> о
    "ύ": "у",  # ύ -> у
    # Greek uppercase -> Cyrillic
    "Α": "А",  # Α -> А
    "Β": "В",  # Β -> В
    "Ε": "Е",  # Ε -> Е
    "Η": "Н",  # Η -> Н
    "Κ": "К",  # Κ -> К
    "Μ": "М",  # Μ -> М
    "Ο": "О",  # Ο -> О
    "Ρ": "Р",  # Ρ -> Р
    "Τ": "Т",  # Τ -> Т
    "Υ": "У",  # Υ -> У
    "Χ": "Х",  # Χ -> Х
    # Latin lowercase -> Cyrillic
    "a": "а",  # a -> а
    "c": "с",  # c -> с
    "e": "е",  # e -> е
    "o": "о",  # o -> о
    "p": "р",  # p -> р
    "x": "х",  # x -> х
    "y": "у",  # y -> у
    # Latin uppercase -> Cyrillic
    "A": "А",  # A -> А
    "B": "В",  # B -> В
    "C": "С",  # C -> С
    "E": "Е",  # E -> Е
    "H": "Н",  # H -> Н
    "K": "К",  # K -> К
    "M": "М",  # M -> М
    "O": "О",  # O -> О
    "P": "Р",  # P -> Р
    "T": "Т",  # T -> Т
    "X": "Х",  # X -> Х
    "Y": "У",  # Y -> У
}

WORD_RE = re.compile(r"[^\W\d_]+")

def is_cyrillic(char: str) -> bool:
    """Return whether a character belongs to the Cyrillic Unicode block.

    Args:
        char: A single-character string.

    Returns:
        ``True`` if the character is in the Cyrillic block (U+0400–U+04FF).
    """
    return "Ѐ" <= char <= "ӿ"


def normalize_word(word: str) -> str:
    """Restore homoglyphs in a mostly-Cyrillic word to their Cyrillic form.

    A word is only rewritten when Cyrillic is its dominant script; foreign
    look-alikes (Greek/Latin) in it are then remapped via :data:`HOMOGLYPH_MAP`.
    Pure-Cyrillic, pure-Latin, and pure-Greek words are left untouched so that
    legitimate names and terms (``React``, ``μg``, ``π``) are never altered.

    Args:
        word: A maximal run of alphabetic characters.

    Returns:
        The word with confusable characters mapped back to Cyrillic, or the
        original word when it is not dominantly Cyrillic.
    """
    cyrillic_count = sum(1 for char in word if is_cyrillic(char))
    if cyrillic_count == 0:
        return word
    foreign_count = sum(1 for char in word if char.isalpha() and not is_cyrillic(char))
    if foreign_count == 0 or cyrillic_count < foreign_count:
        return word
    return "".join(HOMOGLYPH_MAP.get(char, char) for char in word)


def normalize_homoglyphs(text: str) -> str:
    """Repair Greek/Latin homoglyphs spliced into Russian Cyrillic words.

    The models occasionally emit look-alike glyphs (e.g. Greek ``μ``/``ά``)
    inside otherwise-Cyrillic words. This rewrites each such word deterministically
    without any LLM call; residue with no Cyrillic twin is left for the
    foreign-script re-prompt to catch.

    Args:
        text: Arbitrary reply text.

    Returns:
        Text with confusable characters inside mixed Cyrillic words normalized.
    """
    if not text:
        return text
    return WORD_RE.sub(lambda match: normalize_word(match.group(0)), text)


def has_mixed_greek(text: str) -> bool:
    """Return whether any word mixes Cyrillic with Greek characters.

    This is the homoglyph-garbling signature: a Greek glyph spliced into an
    otherwise-Cyrillic word. Standalone Greek symbols (``π``, ``μg``) do not
    match, so legitimate usage is not flagged.

    Args:
        text: Text to inspect (normally already homoglyph-normalized).

    Returns:
        ``True`` if a Cyrillic-and-Greek mixed word is present.
    """
    for word in WORD_RE.findall(text or ""):
        if GREEK_RE.search(word) and any(is_cyrillic(char) for char in word):
            return True
    return False


def needs_russian_correction(text: str) -> bool:
    """Return whether text should be re-prompted for a clean Russian reply.

    Combines the hard-foreign-script signal (CJK, Hangul, Thai, Arabic, Hebrew —
    foreign anywhere) with the mixed-Greek garbling signal (Greek only when fused
    into a Cyrillic word).

    Args:
        text: Reply text to inspect (normally already homoglyph-normalized).

    Returns:
        ``True`` when a re-prompt is warranted.
    """
    if not text:
        return False
    return bool(FOREIGN_SCRIPT_RE.search(text)) or has_mixed_greek(text)


async def apply_language_correction(llm, ai_message, messages: list):
    """Normalize homoglyphs and, if foreign script remains, retry in Russian.

    First applies the deterministic :func:`normalize_homoglyphs` pass. Only when
    the normalized text still contains foreign script does it re-prompt the model
    for a clean Russian reply, then normalizes that result too.

    Args:
        llm: Language model with an async ``ainvoke`` method.
        ai_message: Original AI response to inspect.
        messages: Full message history used for the correction call.

    Returns:
        AI message whose content is normalized, corrected in Russian when needed.
    """
    ai_message.content = normalize_homoglyphs(ai_message.content or "")
    visible = strip_thinking(ai_message.content)
    if not visible or not needs_russian_correction(visible):
        return ai_message
    logger.warning("Foreign script detected, retrying in Russian")
    correction = messages + [HumanMessage(content=LANGUAGE_CORRECTION_PROMPT)]
    try:
        corrected = await llm.ainvoke(correction)
    except Exception as err:
        logger.warning("Language correction failed: %s", err, exc_info=True)
        return ai_message
    corrected.content = normalize_homoglyphs(corrected.content or "")
    return corrected
