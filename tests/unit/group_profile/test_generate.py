"""Group-profile verdict generation tests.

Covers the generation pipeline in isolation, with the LLM round-trip
(``call_profile_model``) mocked:

  - generate_verdicts     : anonymise by user_id, parse {verdict, reason} JSON, remap back
  - fill_missing_verdicts : LLM-omitted members are retried once, then get a neutral fallback
  - build_user_content    : the rubric is delimited and never merged into member data
"""

import json
from unittest.mock import AsyncMock, patch

from src.agent.roast_material import MemberMaterial
from src.group_profile import generate

CALL_MODEL = "src.group_profile.generate.call_profile_model"


def model_json(mapping: dict[str, dict]) -> str:
    """Serialise an anon-keyed {verdict, reason} mapping the way the LLM returns it."""
    return json.dumps(mapping, ensure_ascii=False)


def material(username: str, fact: str = "факт") -> MemberMaterial:
    return MemberMaterial(username=username, facts=[fact])


class TestGenerateVerdicts:
    async def test_anonymises_and_remaps_back_to_user_ids(self):
        materials = {10: material("alice"), 20: material("bob")}
        payload = model_json({
            "user_0": {"verdict": "Циклоп", "reason": "видит только вперёд"},
            "user_1": {"verdict": "Магнето", "reason": "притягивает конфликты"},
        })
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value=payload) as mock_call:
            result = await generate.generate_verdicts("роли из Людей Икс", materials)

        sent = mock_call.await_args[0][1]
        assert "user_0" in sent and "10" not in sent
        assert result[10]["verdict"] == "Циклоп"
        assert result[20]["verdict"] == "Магнето"

    async def test_malformed_json_returns_empty(self):
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value="не json вовсе"):
            result = await generate.generate_verdicts("тема", {10: material("alice")})
        assert result == {}

    async def test_empty_materials_skips_model_call(self):
        with patch(CALL_MODEL, new_callable=AsyncMock) as mock_call:
            result = await generate.generate_verdicts("тема", {})
        assert result == {}
        mock_call.assert_not_awaited()

    async def test_verdict_truncated_to_max_chars(self):
        long_verdict = "Очень длинный вердикт который сильно превышает установленный лимит символов"
        payload = model_json({"user_0": {"verdict": long_verdict, "reason": "x"}})
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value=payload):
            result = await generate.generate_verdicts("тема", {10: material("alice")})
        assert len(result[10]["verdict"]) <= generate.GROUP_PROFILE_VERDICT_MAX_CHARS


class TestBuildUserContent:
    def test_rubric_is_delimited_and_labelled(self):
        content = generate.build_user_content(
            "оцени всем счастье", {10: material("alice")}, {10: "user_0"}
        )
        assert "«оцени всем счастье»" in content
        assert "user_0" in content

    def test_includes_a_dossier_line_per_member(self):
        materials = {10: material("alice"), 20: material("bob")}
        uid_to_anon = {10: "user_0", 20: "user_1"}
        content = generate.build_user_content("тема", materials, uid_to_anon)
        assert "user_0" in content
        assert "user_1" in content


class TestFillMissingVerdicts:
    async def test_omitted_member_recovered_on_reask(self):
        eligible = [1, 2, 3]
        materials = {1: material("a"), 2: material("b"), 3: material("c")}
        verdicts_so_far = {1: {"verdict": "A", "reason": "x"}, 2: {"verdict": "B", "reason": "y"}}
        reask = model_json({"user_0": {"verdict": "Найденный", "reason": "z"}})
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value=reask):
            result = await generate.fill_missing_verdicts(eligible, "тема", materials, verdicts_so_far)
        assert result[3]["verdict"] == "Найденный"

    async def test_still_missing_member_gets_fallback(self):
        eligible = [1, 2, 3]
        materials = {1: material("a"), 2: material("b"), 3: material("c")}
        verdicts_so_far = {1: {"verdict": "A", "reason": "x"}, 2: {"verdict": "B", "reason": "y"}}
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value="{}"):
            result = await generate.fill_missing_verdicts(eligible, "тема", materials, verdicts_so_far)
        assert result[3]["verdict"] == generate.FALLBACK_VERDICT
        assert result[3]["reason"]

    async def test_no_missing_skips_reask(self):
        eligible = [1, 2]
        materials = {1: material("a"), 2: material("b")}
        verdicts_so_far = {1: {"verdict": "A", "reason": "x"}, 2: {"verdict": "B", "reason": "y"}}
        with patch(CALL_MODEL, new_callable=AsyncMock) as mock_call:
            result = await generate.fill_missing_verdicts(eligible, "тема", materials, verdicts_so_far)
        mock_call.assert_not_awaited()
        assert result == verdicts_so_far
