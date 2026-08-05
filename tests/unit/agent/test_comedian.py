"""
Part 4 — ComedianAgent (the autonomous-humor decision brain).

The model returns a strict JSON decision; the value of the feature is that the
parser is fail-safe to *silence*. These tests pin the contract: valid acts
parse, anything malformed/empty/foreign abstains, register is validated, output
is trimmed, and the agent surfaces the parsed decision (or abstains on junk).
"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agent.comedian import (
    ComedianAgent,
    ComedianDecision,
    build_comedian_prompt,
    parse_decision,
)


class TestParseDecision:
    def test_valid_act_decision(self):
        raw = '{"act": true, "register": "light", "text": "Кто из вас тут главный по доте?"}'
        decision = parse_decision(raw)
        assert decision.act is True
        assert decision.register == "light"
        assert "доте" in decision.text

    def test_act_false_abstains(self):
        assert parse_decision('{"act": false, "register": "light", "text": ""}').act is False

    def test_malformed_json_abstains(self):
        assert parse_decision("не json вообще").act is False

    def test_empty_text_with_act_true_abstains(self):
        assert parse_decision('{"act": true, "register": "roast", "text": "   "}').act is False

    def test_invalid_register_defaults_to_light(self):
        raw = '{"act": true, "register": "savage", "text": "Честный топ: кто тут самый токсик?"}'
        decision = parse_decision(raw)
        assert decision.act is True
        assert decision.register == "light"

    def test_extracts_json_from_surrounding_prose(self):
        raw = 'Вот ответ: {"act": true, "register": "roast", "text": "Опять слил катку, да?"} всё'
        assert parse_decision(raw).act is True

    def test_strips_thinking_block(self):
        raw = '<think>стоит ли</think>{"act": true, "register": "light", "text": "Ну и кто ты после этого?"}'
        assert parse_decision(raw).act is True

    def test_trims_to_two_sentences(self):
        raw = '{"act": true, "register": "light", "text": "Раз. Два. Три. Четыре."}'
        assert parse_decision(raw).text.count(".") <= 2

    def test_foreign_script_text_abstains(self):
        assert parse_decision('{"act": true, "register": "light", "text": "这是一个笑话"}').act is False

    def test_latin_meme_words_are_allowed(self):
        raw = '{"act": true, "register": "light", "text": "Это фиаско, братан. ALARM!"}'
        assert parse_decision(raw).act is True


class TestComedianDecisionAbstain:
    def test_abstain_factory(self):
        decision = ComedianDecision.abstain()
        assert decision.act is False
        assert decision.text == ""


class TestBuildComedianPrompt:
    def test_includes_conversation_and_material(self):
        prompt = build_comedian_prompt("@vasya: я лучший", "Факты:\n- проигрывает всем")
        assert "я лучший" in prompt
        assert "проигрывает всем" in prompt
        assert "JSON" in prompt

    def test_omits_material_section_when_empty(self):
        prompt = build_comedian_prompt("@vasya: привет", "")
        assert "Что известно об участниках" not in prompt


class TestComedianAgentDecide:
    async def test_returns_parsed_decision(self):
        agent = ComedianAgent(comedian_executor=object())
        fake = {"messages": [AIMessage(
            content='{"act": true, "register": "light", "text": "Кто проспорил вчера?"}'
        )]}
        with patch("src.agent.comedian.guarded_ainvoke", AsyncMock(return_value=fake)):
            decision = await agent.decide("@vasya: я выиграл", "")
        assert decision.act is True
        assert "проспорил" in decision.text

    async def test_junk_output_abstains(self):
        agent = ComedianAgent(comedian_executor=object())
        fake = {"messages": [AIMessage(content="лол не знаю")]}
        with patch("src.agent.comedian.guarded_ainvoke", AsyncMock(return_value=fake)):
            decision = await agent.decide("@vasya: скучно", "")
        assert decision.act is False

    async def test_raises_when_not_initialised(self):
        with pytest.raises(RuntimeError):
            await ComedianAgent().decide("conv", "mat")
