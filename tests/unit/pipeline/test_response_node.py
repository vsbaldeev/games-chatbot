"""
ResponseNode tests.

Scenario from 4a6cd4e: the full enriched prompt (including reply chains,
recent history, worker output) was stored as the human turn in thread_history.
When the user then replied to a different message, the LLM saw the old enriched
context as the most recent human message and responded about the wrong thread.

Fix: only "@username: user_input" is stored — context is assembled fresh each turn.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import ContextLengthError, DailyLimitError, RateLimitError
from src.config.prompts import (
    LINK_REPLY_GROUNDING_INSTRUCTION,
    SHORTS_TRIGGER_INSTRUCTION,
    SOCIAL_LINK_RETELL_INSTRUCTION,
)
from src.pipeline.response_node import (
    ResponseNode,
    build_asking_user_tag_lines,
    build_directive_lines,
    build_recent_history_lines,
    build_response_input,
    build_trigger_line,
    resolve_group_profile_directive,
    resolve_meme_directive,
)
from tests.builders import make_incoming, make_state

THREAD_GET_HISTORY = "src.pipeline.response_node.thread_history.get_history"
THREAD_APPEND_TURN = "src.pipeline.response_node.thread_history.append_turn"


def make_mock_agent(response_text: str = "Это ответ бота.") -> MagicMock:
    """Return a mock Agent whose invoke_response returns response_text."""
    agent = MagicMock()
    agent.invoke_response = AsyncMock(return_value=response_text)
    return agent


class TestThreadHistoryStorage:
    async def test_stored_human_turn_is_only_username_and_text(self):
        """
        Regression (4a6cd4e): history stored the full enriched prompt with embedded
        reply chains. Next turn the LLM read that as fresh context and answered
        about the wrong thread.  Only the bare user utterance must be persisted.
        """
        agent = make_mock_agent()

        incoming = make_incoming(
            username="alice",
            raw_text="что в этом фото?",
            processed_text="что в этом фото?",
        )
        state = make_state(
            incoming,
            should_respond=True,
            thread_id="thread-42",
            context={
                "user_facts": {},
                "recent_history": [{"message_id": 1, "username": "bob", "content": "hi", "media_type": "text", "user_id": 2}],
                "replied_to": {"message_id": 1, "username": "bob", "content": "hi", "media_type": "text", "user_id": 2},
                "reply_chain": [{"message_id": 1, "username": "bob", "content": "hi", "media_type": "text", "user_id": 2}],
            },
            worker_output="[Данные]: некая информация об игре",
        )

        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock) as mock_append,
        ):
            response_node = ResponseNode(agent)
            await response_node(state)

        mock_append.assert_called_once()
        stored_human_content = mock_append.call_args.kwargs["human_content"]

        assert stored_human_content == "@alice: что в этом фото?"
        assert "reply_chain" not in stored_human_content
        assert "Собранные данные" not in stored_human_content
        assert "Недавние сообщения" not in stored_human_content
        assert "worker_output" not in stored_human_content

    async def test_stored_ai_turn_is_stripped_of_markdown(self):
        """Markdown in AI responses must be stripped before storage to prevent
        the LLM from reinforcing its own markdown formatting in future turns."""
        agent = make_mock_agent(response_text="**Жирный** и _курсив_ текст.")

        incoming = make_incoming(username="bob", raw_text="расскажи", processed_text="расскажи")
        state = make_state(
            incoming,
            should_respond=True,
            thread_id="thread-99",
            context={"user_facts": {}, "recent_history": [], "replied_to": None, "reply_chain": []},
            worker_output="",
        )

        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock) as mock_append,
        ):
            response_node = ResponseNode(agent)
            await response_node(state)

        stored_ai_content = mock_append.call_args.kwargs["ai_content"]
        assert "**" not in stored_ai_content
        assert stored_ai_content == "Жирный и курсив текст."


class TestThinkingBlockStripping:
    async def test_response_text_passed_through_from_agent(self):
        """ResponseNode places whatever invoke_response returns into state['response']
        without modification — thinking stripping is Agent's responsibility."""
        agent = make_mock_agent(response_text="Через 191 день.")

        incoming = make_incoming(
            username="bob",
            raw_text="через сколько дней GTA 6?",
            processed_text="через сколько дней GTA 6?",
        )
        state = make_state(
            incoming,
            should_respond=True,
            thread_id="thread-42",
            context={"user_facts": {}, "recent_history": [], "replied_to": None, "reply_chain": []},
            worker_output="GTA 6 выходит 19 ноября 2026.",
        )

        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            result = await ResponseNode(agent)(state)

        assert result["response"] == "Через 191 день."


class TestRandomTriggerContext:
    """Random triggers get a thin recent-history slice (RANDOM_TRIGGER_CONTEXT_LIMIT).

    Root cause (46013): random photo trigger injected the full recent chat
    history into the response prompt, so the LLM addressed multiple open threads
    and broke the 'never suggest commands unprompted' rule. The revised contract
    keeps the newest few messages so the model can catch obvious topic mismatch
    without turning a spontaneous reaction into a reply to the discussion.
    """

    RECENT_MSG = {
        "message_id": 1,
        "username": "bob",
        "content": "я в мобильном",
        "media_type": "text",
        "user_id": 2,
    }

    def make_context(self) -> dict:
        return {
            "user_facts": {},
            "recent_history": [self.RECENT_MSG],
            "replied_to": None,
            "reply_chain": [],
        }

    def test_random_trigger_gets_reduced_history_slice(self):
        """Random triggers see only the newest RANDOM_TRIGGER_CONTEXT_LIMIT
        messages — older chat noise must stay out of the prompt."""
        recent_newest_first = [
            {
                "message_id": index,
                "username": "bob",
                "content": f"сообщение номер {index}",
                "media_type": "text",
                "user_id": 2,
            }
            for index in range(1, 5)
        ]
        context = {
            "user_facts": {},
            "recent_history": recent_newest_first,
            "replied_to": None,
            "reply_chain": [],
        }

        result = build_response_input(
            "tmaxims",
            "Изображение: руководство по стрижкам",
            "",
            context,
            response_trigger="random",
        )

        assert "сообщение номер 1" in result
        assert "сообщение номер 3" in result
        assert "сообщение номер 4" not in result

    def test_explicit_trigger_includes_recent_history(self):
        """When the user explicitly @mentioned the bot or replied to it, recent
        history must still be included for conversational context."""
        result = build_response_input(
            "alice",
            "@bot что нового?",
            "",
            self.make_context(),
            response_trigger="explicit",
        )

        assert "я в мобильном" in result

    def test_random_trigger_still_includes_user_facts(self):
        """Per-user memories must always be injected — they personalise the
        response regardless of how the trigger fired."""
        context = {
            "user_facts": {"alice": ["любит PS5", "играет в FIFA"]},
            "recent_history": [self.RECENT_MSG],
            "replied_to": None,
            "reply_chain": [],
        }

        result = build_response_input(
            "alice",
            "Изображение: что-то",
            "",
            context,
            response_trigger="random",
        )

        assert "любит PS5" in result

    def test_random_trigger_includes_worker_output(self):
        """Worker facts must always reach the response LLM even for random triggers."""
        result = build_response_input(
            "alice",
            "Изображение: стрижки",
            "Руководство содержит 12 стилей.",
            self.make_context(),
            response_trigger="random",
        )

        assert "Руководство содержит 12 стилей." in result


class TestAskingUserTagLines:
    """The asker's weekly role + reason must be injected so the bot can explain it;
    nothing is emitted when the asker has no role."""

    def test_emits_role_and_reason_when_present(self):
        context = {
            "user_facts": {},
            "recent_history": [],
            "replied_to": None,
            "reply_chain": [],
            "asking_user_tag": {"tag": "Ночной дозор", "reason": "пишет после полуночи"},
        }
        lines = build_asking_user_tag_lines(context, "alice")
        joined = "\n".join(lines)
        assert "Ночной дозор" in joined
        assert "пишет после полуночи" in joined
        assert "alice" in joined

    def test_emits_nothing_when_absent(self):
        context = {"user_facts": {}, "recent_history": [], "replied_to": None, "reply_chain": []}
        assert build_asking_user_tag_lines(context, "alice") == []

    def test_build_response_input_includes_role_block(self):
        context = {
            "user_facts": {},
            "recent_history": [],
            "replied_to": None,
            "reply_chain": [],
            "asking_user_tag": {"tag": "Спидранер", "reason": "проходит за день"},
        }
        result = build_response_input(
            "alice", "почему у меня такая роль?", "", context, response_trigger="explicit"
        )
        assert "Спидранер" in result
        assert "проходит за день" in result


class TestErrorPropagation:
    """Typed exceptions raised by agent.invoke_response must propagate through
    ResponseNode so the top-level handler can surface them to the user."""

    def make_state_for_error(self):
        incoming = make_incoming(username="bob", raw_text="вопрос", processed_text="вопрос")
        return make_state(
            incoming,
            should_respond=True,
            context={"user_facts": {}, "recent_history": [], "replied_to": None, "reply_chain": []},
            worker_output="",
        )

    async def test_context_length_error_propagates(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(side_effect=ContextLengthError("too long"))
        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            with pytest.raises(ContextLengthError):
                await ResponseNode(agent)(self.make_state_for_error())

    async def test_daily_limit_error_propagates(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(side_effect=DailyLimitError("quota"))
        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            with pytest.raises(DailyLimitError):
                await ResponseNode(agent)(self.make_state_for_error())

    async def test_rate_limit_error_propagates(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(side_effect=RateLimitError("rate limit"))
        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            with pytest.raises(RateLimitError):
                await ResponseNode(agent)(self.make_state_for_error())

    async def test_unknown_exception_propagates_unchanged(self):
        agent = MagicMock()
        agent.invoke_response = AsyncMock(side_effect=RuntimeError("unexpected failure"))
        with (
            patch(THREAD_GET_HISTORY, new_callable=AsyncMock, return_value=[]),
            patch(THREAD_APPEND_TURN, new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="unexpected failure"):
                await ResponseNode(agent)(self.make_state_for_error())


class TestLinkRetellRecentHistory:
    """A link retell must see no prior chat history at all.

    Observed in production: two Reels posted back to back made the bot
    describe both in the second caption («Reel 1 – …, Reel 2 – …»). The
    first Reel's ingested material sat in recent history under the same
    ``[Instagram Reel]`` label as the current one, and the retell
    instruction («опирайся только на материалы ниже») gave the model no way
    to tell the two blocks apart.
    """

    PREVIOUS_REEL_ROW = {
        "message_id": 1,
        "username": "tmaxims",
        "user_id": 2,
        "media_type": "text",
        "content": (
            "https://www.instagram.com/reel/Db28QVvuzLC/\n\n"
            "[Instagram Reel]\nэпичный кадр из «Ведьмака» на PS5\n"
            "[Топ-комментарии]:\n- (12 лайков) музыка божественна"
        ),
    }

    @pytest.mark.parametrize("response_trigger", ["social_link", "youtube_short"])
    def test_link_retell_renders_no_recent_history(self, response_trigger):
        context = {"recent_history": [self.PREVIOUS_REEL_ROW]}
        lines, _ = build_recent_history_lines(
            context, response_trigger, has_thread_history=False
        )
        assert lines == []

    def test_link_retell_still_renders_replied_to_material(self):
        """Suppressing history must not take the grounded replied-to block
        with it — that block is about the message being answered, not noise
        from an unrelated earlier link."""
        replied_to = {
            "message_id": 7,
            "username": "bot",
            "user_id": 1,
            "media_type": "video",
            "content": "пересказ прошлого ролика",
            "link_material": "[Instagram Reel]\nматериал прошлого ролика",
        }
        context = {"recent_history": [self.PREVIOUS_REEL_ROW], "replied_to": replied_to}
        lines, returned_replied_to = build_recent_history_lines(
            context, "social_link", has_thread_history=False
        )
        assert returned_replied_to is replied_to
        assert "материал прошлого ролика" in "\n".join(lines)


class TestSocialLinkTriggerFraming:
    def test_social_link_always_uses_retell_framing(self):
        line = build_trigger_line(
            "vasya", "содержимое", "text", None, response_trigger="social_link",
        )
        assert SOCIAL_LINK_RETELL_INSTRUCTION in line

    def test_social_link_framing_names_the_sender(self):
        line = build_trigger_line(
            "vasya", "содержимое", "text", None, response_trigger="social_link",
        )
        assert line.startswith("@vasya")


class TestShortsTriggerFraming:
    def test_shorts_always_uses_retell_framing(self):
        line = build_trigger_line(
            "vasya", "содержимое", "text", None, response_trigger="youtube_short",
        )
        assert SHORTS_TRIGGER_INSTRUCTION in line


class TestCaptionBudgetIsInPrompts:
    def test_shorts_instruction_states_the_character_budget(self):
        assert "600 символов" in SHORTS_TRIGGER_INSTRUCTION

    def test_social_link_instruction_states_the_character_budget(self):
        assert "600 символов" in SOCIAL_LINK_RETELL_INSTRUCTION


class TestLinkReplyGroundingPromptText:
    def test_grounding_instruction_names_the_no_fabrication_rule(self):
        assert "не выдумывай детали" in LINK_REPLY_GROUNDING_INSTRUCTION
        assert "не проверяй факты по своим знаниям" in LINK_REPLY_GROUNDING_INSTRUCTION

    def test_grounding_instruction_has_a_material_placeholder(self):
        rendered = LINK_REPLY_GROUNDING_INSTRUCTION.format(material="ТЕСТ")
        assert "ТЕСТ" in rendered


class TestVoiceTriggerLineFraming:
    def test_voice_is_framed_as_speech_not_description(self):
        line = build_trigger_line("alice", "может не только до попадает", "voice", None)
        assert "сказал голосовым" in line
        assert "не дословные слова автора" not in line

    def test_voice_carries_the_transcript(self):
        line = build_trigger_line("alice", "привет, как дела", "voice", None)
        assert "привет, как дела" in line

    def test_low_confidence_voice_gets_a_softening_note(self):
        line = build_trigger_line(
            "alice", "может не только до попадает", "voice", None,
            voice_low_confidence=True,
        )
        assert "неточ" in line or "разобрал" in line

    def test_default_voice_confidence_is_high_no_softening_note(self):
        line = build_trigger_line("alice", "привет", "voice", None)
        assert "неточ" not in line


class TestMediaTriggerLineJokeIsConditional:
    def test_photo_no_longer_mandates_exaggeration(self):
        line = build_trigger_line("alice", "банка лимонада на столе", "photo", None)
        assert "можно и нужно" not in line

    def test_photo_still_frames_as_description_to_react_to(self):
        line = build_trigger_line("alice", "банка лимонада на столе", "photo", None)
        assert "прислал фото" in line
        assert "банка лимонада на столе" in line

    def test_photo_framing_tells_model_to_answer_a_caption(self):
        line = build_trigger_line(
            "alice", "банка лимонада на столе\n(подпись: рекомендую)", "photo", None,
        )
        assert "отвечай на неё по существу" in line


class TestBuildResponseInputVoiceConfidence:
    def test_voice_low_confidence_flows_through_to_trigger_line(self):
        enriched = build_response_input(
            "alice", "может не только до попадает", "", None,
            media_type="voice", voice_low_confidence=True,
        )
        assert "неточ" in enriched or "разобрал" in enriched


class TestBuildDirectiveLinesInsultReplyingToBot:
    def test_unprovoked_insult_gets_the_comeback(self):
        lines = build_directive_lines(is_bot_insult=True, wind_down=False, photo_directive=None)
        joined = "\n".join(lines)
        assert "дерзкой" in joined
        assert "надоел" not in joined

    def test_insult_replying_to_bot_gets_wind_down_not_comeback(self):
        """filter_node sets wind_down=True alongside is_bot_insult=True when the
        insult replies to the bot's own message, or when the sender's
        engagement tier has already dropped — both cases filter_node treats
        the same way: a mirrored counter-insult would just fuel the loop, so
        only the softer wind-down line should reach the model, never both."""
        lines = build_directive_lines(is_bot_insult=True, wind_down=True, photo_directive=None)
        joined = "\n".join(lines)
        assert "дерзкой" not in joined
        assert "надоел" in joined

    def test_plain_wind_down_unaffected(self):
        lines = build_directive_lines(is_bot_insult=False, wind_down=True, photo_directive=None)
        joined = "\n".join(lines)
        assert "надоел" in joined
        assert "дерзкой" not in joined


class TestMemeRequestSkipsGeneration:
    async def test_accepted_meme_request_returns_an_empty_response(self):
        """The meme is the whole reply — no LLM call, no text to stack on it."""
        agent = make_mock_agent("этого не должно быть")
        state = make_state(make_incoming(), meme_request=True)
        result = await ResponseNode(agent)(state)
        assert result == {"response": "", "response_messages": []}
        agent.invoke_response.assert_not_called()

    async def test_ordinary_message_still_generates(self):
        agent = make_mock_agent("обычный ответ")
        state = make_state(make_incoming(), is_flat_thread=True)
        with patch(THREAD_APPEND_TURN, AsyncMock()):
            result = await ResponseNode(agent)(state)
        assert result["response"] == "обычный ответ"


class TestGroupProfileRequestSkipsGeneration:
    async def test_accepted_group_profile_request_returns_an_empty_response(self):
        """The generated profile is the whole reply — no LLM call, no text to stack on it."""
        agent = make_mock_agent("этого не должно быть")
        state = make_state(make_incoming(), group_profile_request=True)
        result = await ResponseNode(agent)(state)
        assert result == {"response": "", "response_messages": []}
        agent.invoke_response.assert_not_called()


class TestResolveMemeDirective:
    def test_wound_down_meme_request_is_refused(self):
        state = make_state(make_incoming(), wind_down=True, filter_verdict="MEME_REQUEST")
        assert resolve_meme_directive(state) == "refused"

    def test_accepted_meme_request_has_no_directive(self):
        state = make_state(make_incoming(), meme_request=True)
        assert resolve_meme_directive(state) is None

    def test_wound_down_other_verdict_has_no_meme_directive(self):
        state = make_state(make_incoming(), wind_down=True, filter_verdict="BANTER")
        assert resolve_meme_directive(state) is None


class TestResolveGroupProfileDirective:
    def test_wound_down_group_profile_request_is_refused(self):
        state = make_state(make_incoming(), wind_down=True, filter_verdict="GROUP_PROFILE_REQUEST")
        assert resolve_group_profile_directive(state) == "refused"

    def test_accepted_group_profile_request_has_no_directive(self):
        state = make_state(make_incoming(), group_profile_request=True)
        assert resolve_group_profile_directive(state) is None

    def test_wound_down_other_verdict_has_no_group_profile_directive(self):
        state = make_state(make_incoming(), wind_down=True, filter_verdict="BANTER")
        assert resolve_group_profile_directive(state) is None


class TestBuildDirectiveLinesMemeRefusal:
    def test_refusal_line_forbids_inventing_a_meme(self):
        lines = build_directive_lines(
            is_bot_insult=False, wind_down=True, photo_directive=None,
            meme_directive="refused",
        )
        joined = "\n".join(lines)
        assert "мем" in joined
        assert "не выдумывай" in joined

    def test_no_meme_directive_adds_no_meme_line(self):
        lines = build_directive_lines(
            is_bot_insult=False, wind_down=False, photo_directive=None, meme_directive=None,
        )
        assert "мем" not in "\n".join(lines)


class TestBuildDirectiveLinesGroupProfileRefusal:
    def test_refusal_line_forbids_inventing_verdicts(self):
        lines = build_directive_lines(
            is_bot_insult=False, wind_down=True, photo_directive=None,
            group_profile_directive="refused",
        )
        joined = "\n".join(lines)
        assert "роле" in joined or "вердикт" in joined
        assert "не придумывай" in joined.lower()

    def test_no_group_profile_directive_adds_no_group_profile_line(self):
        lines = build_directive_lines(
            is_bot_insult=False, wind_down=False, photo_directive=None,
            group_profile_directive=None,
        )
        assert "вердикт" not in "\n".join(lines)


class TestLinkReplyGrounding:
    def test_material_is_injected_after_the_replied_to_block(self):
        context = {
            "replied_to": {
                "message_id": 5, "user_id": 42, "username": "zhora",
                "content": "Про котиков.", "media_type": "video",
                "link_material": "[YouTube Shorts] транскрипт и кадры",
            },
        }
        lines, _ = build_recent_history_lines(context, "explicit", has_thread_history=False)
        joined = "\n".join(lines)
        assert "[Материал по видео, которое ты запостил]:" in joined
        assert "транскрипт и кадры" in joined
        assert joined.index("Сообщение, на которое отвечают:") < joined.index("[Материал")

    def test_no_material_key_injects_nothing(self):
        context = {
            "replied_to": {
                "message_id": 5, "user_id": 42, "username": "zhora",
                "content": "Обычный ответ.", "media_type": "text",
            },
        }
        lines, _ = build_recent_history_lines(context, "explicit", has_thread_history=False)
        assert not any("Материал по видео" in line for line in lines)

    def test_material_injected_even_when_replied_to_already_shown_in_recent(self):
        row = {
            "message_id": 5, "user_id": 42, "username": "zhora",
            "content": "Про котиков.", "media_type": "video",
            "link_material": "[Instagram Reel] подпись и комментарии",
        }
        context = {"recent_history": [row], "replied_to": row}
        lines, _ = build_recent_history_lines(context, "explicit", has_thread_history=False)
        joined = "\n".join(lines)
        # The row is folded into "Недавние сообщения чата:" and the dedicated
        # "Сообщение, на которое отвечают:" header is skipped — but the
        # material itself was never rendered by that block, so it must still
        # appear.
        assert "Сообщение, на которое отвечают:" not in joined
        assert "подпись и комментарии" in joined
