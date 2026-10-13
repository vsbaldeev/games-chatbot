"""Reaction-emoji sentiment for bot messages.

The operator fixed this set by hand for this specific chat (2026-10-09): these
seven are approval, every other reaction counts as negative. That is a
deliberate over-count, not an oversight — see the design doc's decision 4.

POSITIVE_REACTIONS is stored already normalized (no U+FE0F), because Telegram
sends some reactions with the variation selector and some without. Both the
Python membership test and the rollup SQL compare on the normalized form.
"""

VARIATION_SELECTOR = "️"

POSITIVE_REACTIONS: frozenset[str] = frozenset({
    "❤",        # heart
    "\U0001f44d",    # thumbs up
    "\U0001f602",    # face with tears of joy
    "\U0001f525",    # fire
    "\U0001f4af",    # hundred points
    "\U0001f605",    # grinning face with sweat
    "\U0001f601",    # beaming face with smiling eyes
})


def normalize_reaction(emoji: str) -> str:
    """Strip the variation selector so "❤️" and "❤" compare equal.

    Args:
        emoji: A reaction emoji as Telegram delivered it.

    Returns:
        The same emoji without any U+FE0F codepoints.
    """
    return emoji.replace(VARIATION_SELECTOR, "")


def is_positive_reaction(emoji: str) -> bool:
    """Decide whether a reaction counts as approval.

    Args:
        emoji: A reaction emoji as Telegram delivered it.

    Returns:
        True only for the seven emoji in POSITIVE_REACTIONS; everything
        else, including an empty string, is negative.
    """
    return normalize_reaction(emoji) in POSITIVE_REACTIONS
