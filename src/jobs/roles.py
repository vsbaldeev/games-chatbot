"""Scheduled job: assign personality-based member roles every Sunday afternoon.

The whole pipeline is keyed by ``user_id`` — display names (which fall back to a
non-unique first name) are used only when rendering the announcement. Roles are
decided into a ``user_id``-keyed map first; the announcement is built from that
map, so a member never disappears from the list just because the best-effort
Telegram ``set_chat_member_tag`` call failed. Each role is persisted with a short
reason so the bot can later explain why a member got their role.
"""

import asyncio
import datetime
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from src import achievements, config, log
from src.store import user_tags
from src.store.user_memories import get_facts_for_users

logger = log.get_logger(__name__)

TAG_MODEL = "llama-3.3-70b-versatile"
TAG_MAX_CHARS = 16
MAX_TOKENS = 2048

FALLBACK_ROLE = "Тёмная лошадка"
FALLBACK_REASON = "Пока загадка — фактов маловато, но роль ещё впереди."

SYSTEM_PROMPT = (
    "Ты назначаешь короткие роли участникам чата на основе фактов об их поведении. "
    "Для каждого участника найди самую характерную черту или привычку из фактов — "
    "ту, которая лучше всего его определяет, — и придумай короткую остроумную роль "
    f"на русском языке (строго не длиннее {TAG_MAX_CHARS} символов включая пробелы). "
    "Роль должна быть конкретной и меткой, а не общей. "
    "ВАЖНО: все роли должны быть РАЗНЫМИ — ни одна роль не повторяется у разных участников. "
    "Для каждого участника добавь короткое объяснение (reason) на русском — "
    "одно предложение о том, почему выдана именно эта роль. "
    "Ответь строго в формате JSON: "
    "{\"user_0\": {\"role\": \"роль\", \"reason\": \"объяснение\"}, ...} "
    "используя те же ключи, что и во входных данных. Без какого-либо другого текста."
)


async def call_role_model(system_prompt: str, user_content: str) -> str:
    """Run a single Groq round-trip and return the raw text response.

    Args:
        system_prompt: System instruction for the role model.
        user_content: User turn listing anonymised members and their facts.

    Returns:
        The model's raw response content (expected to be JSON).
    """
    llm = ChatGroq(
        model=TAG_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.5,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        max_retries=0,
    )
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])
    return response.content


def anonymise(user_ids: list[int]) -> tuple[dict[int, str], dict[str, int]]:
    """Map user_ids to opaque ``user_N`` keys so the LLM never sees real ids.

    Args:
        user_ids: User ids to anonymise, in iteration order.

    Returns:
        A ``(uid_to_anon, anon_to_uid)`` pair of inverse mappings.
    """
    uid_to_anon = {user_id: f"user_{index}" for index, user_id in enumerate(user_ids)}
    anon_to_uid = {anon: user_id for user_id, anon in uid_to_anon.items()}
    return uid_to_anon, anon_to_uid


def build_fact_line(anon: str, facts: list[str]) -> str:
    """Render one anonymised member line, e.g. ``user_0: fact1; fact2``."""
    return f"{anon}: {'; '.join(facts)}"


def load_json_object(raw: str) -> dict | None:
    """Strip code fences and parse the LLM response into a dict, or None on failure."""
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, AttributeError) as error:
        logger.warning("Role generation returned non-JSON: %s — %s", error, raw[:200])
        return None
    if not isinstance(data, dict):
        logger.warning("Role generation expected dict, got %s", type(data).__name__)
        return None
    return data


def extract_role_reason(value) -> tuple[str, str]:
    """Pull a trimmed ``(role, reason)`` pair from a JSON value (dict or string)."""
    if isinstance(value, dict):
        role = str(value.get("role") or "").strip()[:TAG_MAX_CHARS]
        reason = str(value.get("reason") or "").strip()
    else:
        role = str(value or "").strip()[:TAG_MAX_CHARS]
        reason = ""
    return role, reason


def parse_roles_response(raw: str, anon_to_uid: dict[str, int]) -> dict[int, dict]:
    """Parse the LLM JSON and remap anon keys back to user_ids.

    Args:
        raw: Raw model response.
        anon_to_uid: Inverse anonymisation mapping for this request.

    Returns:
        Mapping of user_id to ``{"role", "reason"}``; empty on parse failure or
        when no usable roles were produced.
    """
    data = load_json_object(raw)
    if data is None:
        return {}
    roles: dict[int, dict] = {}
    for anon, value in data.items():
        user_id = anon_to_uid.get(anon)
        if user_id is None:
            continue
        role, reason = extract_role_reason(value)
        if role:
            roles[user_id] = {"role": role, "reason": reason}
    return roles


async def generate_roles(facts_by_uid: dict[int, list[str]]) -> dict[int, dict]:
    """Ask the LLM for a role + reason per member, keyed by user_id.

    Args:
        facts_by_uid: Mapping of user_id to that member's behavioural facts.

    Returns:
        Mapping of user_id to ``{"role", "reason"}`` for members the LLM tagged.
    """
    if not facts_by_uid:
        return {}
    uid_to_anon, anon_to_uid = anonymise(list(facts_by_uid))
    lines = [build_fact_line(uid_to_anon[user_id], facts) for user_id, facts in facts_by_uid.items()]
    raw = await call_role_model(SYSTEM_PROMPT, "\n".join(lines))
    return parse_roles_response(raw, anon_to_uid)


def find_duplicate_uids(roles: dict[int, dict]) -> list[int]:
    """Return user_ids whose role repeats one already seen (case-insensitive)."""
    seen: set[str] = set()
    duplicates: list[int] = []
    for user_id, entry in roles.items():
        key = entry["role"].casefold()
        if key in seen:
            duplicates.append(user_id)
        else:
            seen.add(key)
    return duplicates


async def reask_unique(facts_by_uid: dict[int, list[str]], taken_roles: set[str]) -> dict[int, dict]:
    """Re-ask the LLM for distinct roles for the given members, avoiding taken ones."""
    if not facts_by_uid:
        return {}
    uid_to_anon, anon_to_uid = anonymise(list(facts_by_uid))
    lines = [build_fact_line(uid_to_anon[user_id], facts) for user_id, facts in facts_by_uid.items()]
    taken = ", ".join(sorted(taken_roles))
    user_content = (
        "\n".join(lines)
        + f"\n\nЭти роли уже заняты другими участниками: {taken}. "
        "Придумай каждому ДРУГУЮ, уникальную роль, не совпадающую с занятыми."
    )
    raw = await call_role_model(SYSTEM_PROMPT, user_content)
    return parse_roles_response(raw, anon_to_uid)


def make_unique_role(role: str, seen: set[str]) -> str:
    """Return ``role`` unchanged, or a numerically-suffixed variant not in ``seen``."""
    if role.casefold() not in seen:
        return role
    for index in range(2, 100):
        suffix = f" {index}"
        candidate = role[: TAG_MAX_CHARS - len(suffix)].rstrip() + suffix
        if candidate.casefold() not in seen:
            return candidate
    return role[:TAG_MAX_CHARS]


def disambiguate(roles: dict[int, dict]) -> dict[int, dict]:
    """Deterministic safety net: force every role string to be unique."""
    seen: set[str] = set()
    result: dict[int, dict] = {}
    for user_id, entry in roles.items():
        unique_role = make_unique_role(entry["role"], seen)
        seen.add(unique_role.casefold())
        result[user_id] = {**entry, "role": unique_role}
    return result


async def enforce_unique_roles(
    roles: dict[int, dict], facts_by_uid: dict[int, list[str]]
) -> dict[int, dict]:
    """Ensure no two members share a role: re-ask once, then dedup deterministically.

    Args:
        roles: Current user_id to ``{"role", "reason"}`` mapping.
        facts_by_uid: Facts for the colliding members, used for the re-ask.

    Returns:
        A mapping with strictly unique role strings.
    """
    duplicate_uids = find_duplicate_uids(roles)
    if duplicate_uids:
        taken = {roles[user_id]["role"] for user_id in roles if user_id not in duplicate_uids}
        dup_facts = {user_id: facts_by_uid[user_id] for user_id in duplicate_uids if user_id in facts_by_uid}
        for user_id, entry in (await reask_unique(dup_facts, taken)).items():
            roles[user_id] = entry
    return disambiguate(roles)


async def fill_missing_roles(
    eligible_uids: list[int], facts_by_uid: dict[int, list[str]], roles: dict[int, dict]
) -> dict[int, dict]:
    """Recover members the LLM omitted: re-ask once, then assign a neutral fallback.

    Args:
        eligible_uids: All members that should receive a role.
        facts_by_uid: Facts per eligible member.
        roles: Roles produced so far (mutated and returned).

    Returns:
        ``roles`` with an entry for every eligible member.
    """
    missing = [user_id for user_id in eligible_uids if user_id not in roles]
    if not missing:
        return roles
    recovered = await generate_roles({user_id: facts_by_uid[user_id] for user_id in missing})
    roles.update(recovered)
    for user_id in missing:
        roles.setdefault(user_id, {"role": FALLBACK_ROLE, "reason": FALLBACK_REASON})
    return roles


def to_assignments(roles: dict[int, dict]) -> dict[int, dict]:
    """Convert internal role entries into the store's ``{"tag", "reason"}`` shape."""
    return {
        user_id: {"tag": entry["role"], "reason": entry["reason"]}
        for user_id, entry in roles.items()
    }


async def announce_roles(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    roles: dict[int, dict],
    names_by_uid: dict[int, str],
) -> None:
    """Post the weekly roles, built from the decided map (not from API success)."""
    if not roles:
        return
    lines = "\n".join(
        f"@{names_by_uid.get(user_id, user_id)} — {entry['role']}"
        for user_id, entry in roles.items()
    )
    try:
        await context.bot.send_message(chat_id=chat_id, text=f"🏷 Роли недели:\n\n{lines}")
    except Exception as error:
        logger.warning("Failed to announce roles for chat %s: %s", chat_id, error)


async def set_member_tag(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, role: str
) -> None:
    """Best-effort Telegram tag set for one member; never raises."""
    try:
        await context.bot.set_chat_member_tag(chat_id=chat_id, user_id=user_id, tag=role[:TAG_MAX_CHARS])
    except BadRequest as error:
        if "Chat_creator_required" in str(error):
            logger.debug("set_chat_member_tag skipped for chat %s: bot is not creator", chat_id)
        else:
            logger.warning("Failed to set tag for user %s in chat %s: %s", user_id, chat_id, error)
    except Exception as error:
        logger.warning("Failed to set tag for user %s in chat %s: %s", user_id, chat_id, error)


async def apply_telegram_tags(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, roles: dict[int, dict]
) -> None:
    """Apply every decided role as a Telegram member tag (best-effort)."""
    for user_id, entry in roles.items():
        await set_member_tag(context, chat_id, user_id, entry["role"])


async def assign_roles_for_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Decide, persist, announce, and apply weekly roles for one chat.

    Args:
        context: Telegram context used to send messages and set tags.
        chat_id: The chat to process.
    """
    members = await achievements.get_chat_members(chat_id)
    if not members:
        return
    names_by_uid = {user_id: name for user_id, name in members}
    facts_by_uid = await get_facts_for_users(chat_id=chat_id, user_ids=list(names_by_uid))
    eligible = [user_id for user_id in names_by_uid if facts_by_uid.get(user_id)]
    if not eligible:
        return
    eligible_facts = {user_id: facts_by_uid[user_id] for user_id in eligible}

    roles = await generate_roles(eligible_facts)
    roles = await fill_missing_roles(eligible, eligible_facts, roles)
    roles = await enforce_unique_roles(roles, eligible_facts)

    await user_tags.upsert_tags(chat_id=chat_id, assignments=to_assignments(roles))
    await announce_roles(context, chat_id, roles, names_by_uid)
    await apply_telegram_tags(context, chat_id, roles)


async def weekly_roles_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the role assignment for every known chat, but only on Sundays."""
    if datetime.datetime.now(datetime.timezone.utc).weekday() != 6:  # 6 = Sunday
        return
    chat_ids = await achievements.get_all_chat_ids()
    await asyncio.gather(
        *[assign_roles_for_chat(context, chat_id) for chat_id in chat_ids],
        return_exceptions=True,
    )
