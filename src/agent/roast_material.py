"""Gather rich, real material about a chat member for humor/roast prompts.

Pulls a member's facts, recent quotes, weekly role, and most notable stats via
the existing stores and formats them into one compact prompt block. Every
section degrades to empty when its data is absent, so callers can drop the
formatted block straight into a prompt without conditionals.

Shared by the comedian (autonomous humor) and the offense auto-roast — both want
jokes grounded in real specifics rather than distilled facts alone.
"""

from dataclasses import dataclass, field

from src import achievements
from src.store import unified_messages, user_tags
from src.store.user_memories import get_facts

MAX_FACTS = 12
MAX_QUOTES = 3
MAX_QUOTE_CHARS = 160
MAX_STAT_HIGHLIGHTS = 4

# Curated, roast-worthy stats → Russian phrase templates ({count} is filled in).
# Only these keys surface as highlights; the rest of the stat row is ignored.
STAT_LABELS = {
    "roasted_count": "прожарен раз: {count}",
    "duel_wins": "побед в дуэлях: {count}",
    "night_messages": "ночных сообщений: {count}",
    "sticker_messages": "стикеров отправил: {count}",
    "voice_messages": "голосовых записал: {count}",
    "forwarded_messages": "репостов накидал: {count}",
    "link_messages": "ссылок скинул: {count}",
    "photo_messages": "фоток выложил: {count}",
    "laugh_reactions": "поймал 😂-реакций: {count}",
    "fire_reactions": "поймал 🔥-реакций: {count}",
    "long_messages": "простыней написал: {count}",
}


@dataclass
class MemberMaterial:
    """Real, formatted material about a single chat member.

    Attributes:
        username: Member username without ``@``.
        facts: Stored facts, newest-first, capped at ``MAX_FACTS``.
        quotes: Recent verbatim messages, capped in count and length.
        role: ``{tag, reason}`` weekly role, or ``None`` when unassigned.
        stats: Notable stat highlights as short Russian phrases.
    """

    username: str
    facts: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    role: dict | None = None
    stats: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Return True when no material of any kind is available."""
        return not (self.facts or self.quotes or self.role or self.stats)


def select_stat_highlights(stats: dict[str, int]) -> list[str]:
    """Pick the most notable non-zero stats as short Russian phrases.

    Args:
        stats: Full stat row from ``achievements.get_user_stats``.

    Returns:
        Up to ``MAX_STAT_HIGHLIGHTS`` phrases, highest counts first.
    """
    scored = [
        (key, value) for key, value in stats.items()
        if key in STAT_LABELS and value > 0
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        STAT_LABELS[key].format(count=value)
        for key, value in scored[:MAX_STAT_HIGHLIGHTS]
    ]


def truncate_quotes(messages: list[str]) -> list[str]:
    """Trim to a few recent quotes, each capped at ``MAX_QUOTE_CHARS``.

    Args:
        messages: Recent verbatim messages, newest-first.

    Returns:
        Up to ``MAX_QUOTES`` non-blank quotes, long ones ellipsised.
    """
    quotes: list[str] = []
    for message in messages[:MAX_QUOTES]:
        text = message.strip()
        if not text:
            continue
        if len(text) > MAX_QUOTE_CHARS:
            text = text[:MAX_QUOTE_CHARS].rstrip() + "…"
        quotes.append(text)
    return quotes


async def gather_member_material(chat_id: int, user_id: int, username: str) -> MemberMaterial:
    """Collect facts, quotes, role, and notable stats for one member.

    Args:
        chat_id: Group chat the member belongs to.
        user_id: Telegram user id of the member.
        username: Member username without ``@`` (used for quote lookup).

    Returns:
        A populated ``MemberMaterial`` (possibly empty if the member is unknown).
    """
    facts = await get_facts(chat_id=chat_id, user_id=user_id)
    messages = await unified_messages.get_user_messages(
        chat_id=chat_id, username=username, limit=MAX_QUOTES * 4
    )
    role = await user_tags.get_tag(chat_id=chat_id, user_id=user_id)
    stats = await achievements.get_user_stats(user_id, chat_id)
    return MemberMaterial(
        username=username,
        facts=facts[:MAX_FACTS],
        quotes=truncate_quotes(messages),
        role=role,
        stats=select_stat_highlights(stats),
    )


def format_facts(material: MemberMaterial) -> list[str]:
    """Return the facts section lines, or empty when absent."""
    if not material.facts:
        return []
    return ["Факты:", *(f"- {fact}" for fact in material.facts)]


def format_quotes(material: MemberMaterial) -> list[str]:
    """Return the quotes section lines, or empty when absent."""
    if not material.quotes:
        return []
    return ["Что сам писал:", *(f"- «{quote}»" for quote in material.quotes)]


def format_role(material: MemberMaterial) -> list[str]:
    """Return the weekly-role section lines, or empty when absent."""
    role = material.role or {}
    tag = role.get("tag")
    if not tag:
        return []
    lines = [f"Роль недели: {tag}"]
    reason = role.get("reason")
    if reason:
        lines.append(f"За что: {reason}")
    return lines


def format_stats(material: MemberMaterial) -> list[str]:
    """Return the stats section line, or empty when absent."""
    if not material.stats:
        return []
    return ["Статы: " + "; ".join(material.stats)]


def format_member_material(material: MemberMaterial) -> str:
    """Assemble the full compact prompt block for one member.

    Args:
        material: The gathered material.

    Returns:
        A newline-joined block, or an empty string when there is no material.
    """
    lines = [
        *format_facts(material),
        *format_quotes(material),
        *format_role(material),
        *format_stats(material),
    ]
    return "\n".join(lines)
