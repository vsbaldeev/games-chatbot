"""Group-profile verdict generation — one rubric, one LLM call, full coverage.

Mirrors src/jobs/roles.py's anonymise-then-generate-then-fill-missing shape,
but the theme is a free-form user-supplied rubric instead of a fixed "assign
a role" instruction, and there is no uniqueness pass — a scoring rubric
("7/10") has no reason to avoid repeats, so that guarantee stays specific to
weekly roles.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent import ainvoke_with_backoff
from src.agent.language import normalize_homoglyphs
from src.agent.roast_material import MemberMaterial, format_member_material
from src.config.prompts import (
    GROUP_PROFILE_SYSTEM,
    GROUP_PROFILE_VERDICT_MAX_CHARS,
)
from src.utils.anon_map import anonymise
from src.utils.llm_json import load_json_object

logger = log.get_logger(__name__)

MAX_TOKENS = 2048

FALLBACK_VERDICT = "Загадка"
FALLBACK_REASON = "Фактов маловато — тут я пас."


async def call_profile_model(system_prompt: str, user_content: str) -> str:
    """Run a single Groq round-trip and return the raw text response.

    TAG_MODEL is a reasoning model; reasoning_effort="none" (which Groq accepts
    for this model, unlike the previous gpt-oss-120b primary) keeps the whole
    max_tokens budget free for the JSON body itself, so a large roster cannot
    truncate it mid-string the way an unconstrained or "low"-capped reasoning
    pass could (finish_reason="length" — the check below is a backstop, not
    the primary defense). Retries transient TPM 429s via ainvoke_with_backoff,
    since a truncated call re-asks for the full roster in fill_missing_verdicts
    and can trip the per-minute limit on the second call. normalize_homoglyphs
    repairs the occasional Latin/Greek glyph this model splices into an
    otherwise-Cyrillic word (e.g. a role rendered "Синдikat") — the same
    deterministic pass already applied to RESPONSE/ROAST output.

    Args:
        system_prompt: System instruction for the profile model.
        user_content: User turn carrying the rubric and anonymised dossiers.

    Returns:
        The model's raw response content (expected to be JSON).
    """
    llm = ChatGroq(
        model=config.TAG_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        max_retries=0,
        reasoning_effort="none",
    )
    response = await ainvoke_with_backoff(llm, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])
    if response.response_metadata.get("finish_reason") == "length":
        logger.warning(
            "Group profile generation hit max_tokens=%d before the JSON finished; "
            "response will fail to parse",
            MAX_TOKENS,
        )
    return normalize_homoglyphs(response.content)


def build_dossier_line(anon: str, material: MemberMaterial) -> str:
    """Render one anonymised member's dossier block for the prompt."""
    block = format_member_material(material)
    return f"{anon}:\n{block}" if block else f"{anon}: (нет данных)"


def build_user_content(
    rubric: str, materials_by_uid: dict[int, MemberMaterial], uid_to_anon: dict[int, str]
) -> str:
    """Assemble the human turn: the rubric, delimited, followed by each dossier.

    The rubric is the user's own request text, kept verbatim but delimited —
    it is instructed content applied to a fixed schema, never concatenated
    into the system prompt, so it cannot redefine the output format.

    Args:
        rubric: The user's request text.
        materials_by_uid: Dossiers for members eligible for the LLM call.
        uid_to_anon: This request's user_id to anon-key mapping.

    Returns:
        The assembled human-turn string.
    """
    lines = [
        build_dossier_line(uid_to_anon[user_id], material)
        for user_id, material in materials_by_uid.items()
    ]
    return (
        f"Тема (rubric) — применяй буквально, в формулировке участника:\n«{rubric}»\n\n"
        "Участники:\n" + "\n\n".join(lines)
    )


def extract_verdict(value) -> tuple[str, str]:
    """Pull a trimmed ``(verdict, reason)`` pair from a JSON value (dict or string)."""
    if isinstance(value, dict):
        verdict = str(value.get("verdict") or "").strip()[:GROUP_PROFILE_VERDICT_MAX_CHARS]
        reason = str(value.get("reason") or "").strip()
    else:
        verdict = str(value or "").strip()[:GROUP_PROFILE_VERDICT_MAX_CHARS]
        reason = ""
    return verdict, reason


def parse_profile_response(raw: str, anon_to_uid: dict[str, int]) -> dict[int, dict]:
    """Parse the LLM JSON and remap anon keys back to user_ids.

    Args:
        raw: Raw model response.
        anon_to_uid: Inverse anonymisation mapping for this request.

    Returns:
        Mapping of user_id to ``{"verdict", "reason"}``; empty on parse
        failure or when no usable verdicts were produced.
    """
    data = load_json_object(raw, context="Group profile generation")
    if data is None:
        return {}
    verdicts: dict[int, dict] = {}
    for anon, value in data.items():
        user_id = anon_to_uid.get(anon)
        if user_id is None:
            continue
        verdict, reason = extract_verdict(value)
        if verdict:
            verdicts[user_id] = {"verdict": verdict, "reason": reason}
    return verdicts


async def generate_verdicts(rubric: str, materials_by_uid: dict[int, MemberMaterial]) -> dict[int, dict]:
    """Ask the LLM for a verdict + reason per member, keyed by user_id.

    Args:
        rubric: The user's own request text, applied verbatim as the theme.
        materials_by_uid: Dossiers for members eligible for the LLM call.

    Returns:
        Mapping of user_id to ``{"verdict", "reason"}`` for members the LLM covered.
    """
    if not materials_by_uid:
        return {}
    uid_to_anon, anon_to_uid = anonymise(list(materials_by_uid))
    user_content = build_user_content(rubric, materials_by_uid, uid_to_anon)
    raw = await call_profile_model(GROUP_PROFILE_SYSTEM, user_content)
    return parse_profile_response(raw, anon_to_uid)


async def fill_missing_verdicts(
    eligible_uids: list[int],
    rubric: str,
    materials_by_uid: dict[int, MemberMaterial],
    verdicts: dict[int, dict],
) -> dict[int, dict]:
    """Recover members the LLM omitted: re-ask once, then a neutral fallback.

    Args:
        eligible_uids: All members that should receive a verdict.
        rubric: The user's own request text, re-used for the re-ask.
        materials_by_uid: Dossiers for eligible members.
        verdicts: Verdicts produced so far (mutated and returned).

    Returns:
        ``verdicts`` with an entry for every eligible member.
    """
    missing = [user_id for user_id in eligible_uids if user_id not in verdicts]
    if not missing:
        return verdicts
    missing_materials = {user_id: materials_by_uid[user_id] for user_id in missing}
    recovered = await generate_verdicts(rubric, missing_materials)
    verdicts.update(recovered)
    for user_id in missing:
        verdicts.setdefault(user_id, {"verdict": FALLBACK_VERDICT, "reason": FALLBACK_REASON})
    return verdicts
