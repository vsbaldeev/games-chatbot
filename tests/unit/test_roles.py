"""
Weekly roles job tests.

Covers the role-assignment pipeline in isolation, with the LLM round-trip
(``call_role_model``) and all store/Telegram I/O mocked:

  - generate_roles      : anonymise by user_id, parse {role, reason} JSON, remap back
  - enforce_unique_roles: duplicate roles trigger a re-ask, then deterministic dedup
  - fill_missing_roles  : LLM-omitted members are retried, then get a neutral fallback
  - announce_roles      : message built from the decided roles map, not API success
  - apply_telegram_tags : Telegram errors are swallowed, never affect the announcement
  - assign_roles_for_chat: factless members excluded; a member whose tag-set call
                           raises still appears in the announcement (the original bug)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from telegram.error import BadRequest

from src.jobs import roles

CALL_MODEL = "src.jobs.roles.call_role_model"
GET_MEMBERS = "src.jobs.roles.achievements.get_chat_members"
GET_FACTS = "src.jobs.roles.get_facts_for_users"
UPSERT_TAGS = "src.jobs.roles.user_tags.upsert_tags"

CHAT_ID = 1000


def model_json(mapping: dict[str, dict]) -> str:
    """Serialise an anon-keyed {role, reason} mapping the way the LLM returns it."""
    return json.dumps(mapping, ensure_ascii=False)


def make_context() -> MagicMock:
    """Telegram context with awaitable bot send/tag methods."""
    context = MagicMock()
    context.bot.send_message = AsyncMock()
    context.bot.set_chat_member_tag = AsyncMock()
    return context


class TestGenerateRoles:
    async def test_anonymises_and_remaps_back_to_user_ids(self):
        facts = {10: ["играет ночью"], 20: ["проходит игры за день"]}
        payload = model_json({
            "user_0": {"role": "Ночной дозор", "reason": "пишет ночью"},
            "user_1": {"role": "Спидранер", "reason": "очень быстро"},
        })
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value=payload) as mock_call:
            result = await roles.generate_roles(facts)

        # The model must never see real user_ids, only anon keys.
        sent = mock_call.await_args[0][1]
        assert "user_0" in sent and "10" not in sent
        assert result[10]["role"] == "Ночной дозор"
        assert result[10]["reason"] == "пишет ночью"
        assert result[20]["role"] == "Спидранер"

    async def test_malformed_json_returns_empty(self):
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value="не json вовсе"):
            result = await roles.generate_roles({10: ["факт"]})
        assert result == {}

    async def test_empty_facts_skips_model_call(self):
        with patch(CALL_MODEL, new_callable=AsyncMock) as mock_call:
            result = await roles.generate_roles({})
        assert result == {}
        mock_call.assert_not_awaited()

    async def test_role_truncated_to_max_chars(self):
        long_role = "Очень длинная роль которая превышает лимит"
        payload = model_json({"user_0": {"role": long_role, "reason": "x"}})
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value=payload):
            result = await roles.generate_roles({10: ["факт"]})
        assert len(result[10]["role"]) <= roles.TAG_MAX_CHARS


class TestEnforceUniqueRoles:
    async def test_duplicate_triggers_reask_for_distinct_role(self):
        current = {
            1: {"role": "Тег", "reason": "a"},
            2: {"role": "тег", "reason": "b"},     # case-insensitive duplicate of #1
            3: {"role": "Другой", "reason": "c"},
        }
        facts = {1: ["f1"], 2: ["f2"], 3: ["f3"]}
        reask = model_json({"user_0": {"role": "Уникальный", "reason": "b2"}})
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value=reask):
            result = await roles.enforce_unique_roles(current, facts)

        final_roles = [entry["role"].casefold() for entry in result.values()]
        assert len(set(final_roles)) == 3

    async def test_deterministic_dedup_when_reask_still_collides(self):
        current = {
            1: {"role": "Тег", "reason": "a"},
            2: {"role": "Тег", "reason": "b"},
        }
        facts = {1: ["f1"], 2: ["f2"]}
        # The re-ask stubbornly returns the same colliding role.
        reask = model_json({"user_0": {"role": "Тег", "reason": "b2"}})
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value=reask):
            result = await roles.enforce_unique_roles(current, facts)

        final_roles = [entry["role"].casefold() for entry in result.values()]
        assert len(set(final_roles)) == 2
        for entry in result.values():
            assert len(entry["role"]) <= roles.TAG_MAX_CHARS

    async def test_no_duplicates_skips_reask(self):
        current = {1: {"role": "Один", "reason": "a"}, 2: {"role": "Два", "reason": "b"}}
        with patch(CALL_MODEL, new_callable=AsyncMock) as mock_call:
            result = await roles.enforce_unique_roles(current, {1: ["f"], 2: ["g"]})
        mock_call.assert_not_awaited()
        assert result == current


class TestFillMissingRoles:
    async def test_omitted_member_recovered_on_reask(self):
        eligible = [1, 2, 3]
        facts = {1: ["f1"], 2: ["f2"], 3: ["f3"]}
        roles_so_far = {1: {"role": "A", "reason": "x"}, 2: {"role": "B", "reason": "y"}}
        reask = model_json({"user_0": {"role": "Найденный", "reason": "z"}})
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value=reask):
            result = await roles.fill_missing_roles(eligible, facts, roles_so_far)
        assert result[3]["role"] == "Найденный"

    async def test_still_missing_member_gets_fallback(self):
        eligible = [1, 2, 3]
        facts = {1: ["f1"], 2: ["f2"], 3: ["f3"]}
        roles_so_far = {1: {"role": "A", "reason": "x"}, 2: {"role": "B", "reason": "y"}}
        with patch(CALL_MODEL, new_callable=AsyncMock, return_value="{}"):
            result = await roles.fill_missing_roles(eligible, facts, roles_so_far)
        assert result[3]["role"] == roles.FALLBACK_ROLE
        assert result[3]["reason"]

    async def test_no_missing_skips_reask(self):
        eligible = [1, 2]
        facts = {1: ["f1"], 2: ["f2"]}
        roles_so_far = {1: {"role": "A", "reason": "x"}, 2: {"role": "B", "reason": "y"}}
        with patch(CALL_MODEL, new_callable=AsyncMock) as mock_call:
            result = await roles.fill_missing_roles(eligible, facts, roles_so_far)
        mock_call.assert_not_awaited()
        assert result == roles_so_far


class TestAnnounceRoles:
    async def test_lists_every_user_in_roles_map(self):
        context = make_context()
        role_map = {1: {"role": "Ночной", "reason": "r1"}, 2: {"role": "Спидран", "reason": "r2"}}
        names = {1: "alice", 2: "bob"}
        await roles.announce_roles(context, CHAT_ID, role_map, names)

        context.bot.send_message.assert_awaited_once()
        text = context.bot.send_message.await_args.kwargs["text"]
        assert "@alice — Ночной" in text
        assert "@bob — Спидран" in text

    async def test_empty_roles_sends_nothing(self):
        context = make_context()
        await roles.announce_roles(context, CHAT_ID, {}, {})
        context.bot.send_message.assert_not_awaited()


class TestApplyTelegramTags:
    async def test_swallows_chat_creator_required(self):
        context = make_context()
        context.bot.set_chat_member_tag = AsyncMock(
            side_effect=BadRequest("Chat_creator_required")
        )
        role_map = {1: {"role": "Тег", "reason": "r"}}
        # Must not raise.
        await roles.apply_telegram_tags(context, CHAT_ID, role_map)
        context.bot.set_chat_member_tag.assert_awaited_once()

    async def test_sets_tag_for_each_member(self):
        context = make_context()
        role_map = {1: {"role": "A", "reason": "r"}, 2: {"role": "B", "reason": "r"}}
        await roles.apply_telegram_tags(context, CHAT_ID, role_map)
        assert context.bot.set_chat_member_tag.await_count == 2


class TestAssignRolesForChat:
    async def test_member_appears_in_announcement_even_if_tag_set_fails(self):
        """The original bug: a member whose set_chat_member_tag raised was dropped
        from the announcement even though the tag was applied. The announcement
        must now be decoupled from the Telegram call."""
        context = make_context()

        def tag_side_effect(*, chat_id, user_id, tag):
            if user_id == 2:
                raise BadRequest("transient failure")
            return None

        context.bot.set_chat_member_tag = AsyncMock(side_effect=tag_side_effect)
        payload = model_json({
            "user_0": {"role": "Альфа", "reason": "r1"},
            "user_1": {"role": "Бета", "reason": "r2"},
        })
        with patch(GET_MEMBERS, new_callable=AsyncMock, return_value=[(1, "alice"), (2, "bob")]), \
             patch(GET_FACTS, new_callable=AsyncMock, return_value={1: ["f1"], 2: ["f2"]}), \
             patch(UPSERT_TAGS, new_callable=AsyncMock), \
             patch(CALL_MODEL, new_callable=AsyncMock, return_value=payload):
            await roles.assign_roles_for_chat(context, CHAT_ID)

        text = context.bot.send_message.await_args.kwargs["text"]
        assert "@alice" in text
        assert "@bob" in text  # present despite its tag-set call raising

    async def test_factless_members_excluded_and_eligible_persisted(self):
        context = make_context()
        payload = model_json({"user_0": {"role": "Альфа", "reason": "r1"}})
        with patch(GET_MEMBERS, new_callable=AsyncMock, return_value=[(1, "alice"), (2, "bob")]), \
             patch(GET_FACTS, new_callable=AsyncMock, return_value={1: ["f1"]}), \
             patch(UPSERT_TAGS, new_callable=AsyncMock) as mock_upsert, \
             patch(CALL_MODEL, new_callable=AsyncMock, return_value=payload):
            await roles.assign_roles_for_chat(context, CHAT_ID)

        assignments = mock_upsert.await_args.kwargs["assignments"]
        assert set(assignments) == {1}  # bob (factless) excluded
        assert assignments[1]["tag"] == "Альфа"
        assert assignments[1]["reason"] == "r1"

    async def test_no_members_does_nothing(self):
        context = make_context()
        with patch(GET_MEMBERS, new_callable=AsyncMock, return_value=[]), \
             patch(UPSERT_TAGS, new_callable=AsyncMock) as mock_upsert:
            await roles.assign_roles_for_chat(context, CHAT_ID)
        mock_upsert.assert_not_awaited()
        context.bot.send_message.assert_not_awaited()
