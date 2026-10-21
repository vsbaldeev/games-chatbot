"""ClassifyNode tests: LLM classification, reply-context overrides, label parsing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.prompts import FILTER_SYSTEM
from src.pipeline.filter.classify import ClassifyNode, GROUP_PROFILE_MARKER_RE, parse_label
from src.pipeline.router import looks_like_request
from tests.builders import make_incoming, make_state


def make_classify_node(
    llm_response: str | None = None,
    llm_error: Exception | None = None,
) -> tuple[ClassifyNode, MagicMock]:
    mock_llm = MagicMock()
    if llm_error is not None:
        mock_llm.ainvoke = AsyncMock(side_effect=llm_error)
    else:
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content=llm_response or "MEANINGFUL")
        )
    return ClassifyNode(llm=mock_llm, confirm_llm=mock_llm), mock_llm


class TestFilterSystemPrompt:
    def test_mentions_meaningless_label(self):
        assert "MEANINGLESS" in FILTER_SYSTEM

    def test_mentions_meaningful_label(self):
        assert "MEANINGFUL" in FILTER_SYSTEM

    def test_instructs_single_word_reply(self):
        assert "one word" in FILTER_SYSTEM.lower()

    def test_defaults_to_meaningful_when_uncertain(self):
        assert "unsure" in FILTER_SYSTEM.lower()

    def test_mentions_meme_request_label(self):
        assert "MEME_REQUEST" in FILTER_SYSTEM

    def test_no_longer_routes_meme_requests_to_meaningful(self):
        """«скинь мем» used to be named as a MEANINGFUL example by the prompt."""
        assert "'скинь мем', 'кинь мемас'" in FILTER_SYSTEM

    def test_mentions_group_profile_request_label(self):
        assert "GROUP_PROFILE_REQUEST" in FILTER_SYSTEM

    def test_mentions_not_addressed_label(self):
        assert "NOT_ADDRESSED" in FILTER_SYSTEM

    def test_no_longer_asserts_the_message_is_addressed(self):
        """The premise made the addressee question unaskable (2026-08-18 incident)."""
        assert "is addressed to the bot" not in FILTER_SYSTEM

    def test_third_person_talk_is_named_as_not_addressed(self):
        assert "он говорит" in FILTER_SYSTEM

    def test_defaults_away_from_not_addressed_when_unsure(self):
        """Fail-open is preserved: silence is the riskier failure."""
        assert "never 'NOT_ADDRESSED'" in FILTER_SYSTEM


class TestRequestOverrideIgnoresMentions:
    """The question/request override judges what the user typed, not the @handle.

    An addressed message reaches the filter with its «@bot» prefix intact, so
    the handle used to take the leading-word slot and inflate the word count —
    silently disabling the override for every @mentioned short question.
    """

    @pytest.mark.parametrize("text", [
        "@zhora_bot о чем я говорю",
        "@zhora_bot что я описал",
        "@zhora_bot как это работает",
        "@zhora_bot переведи",
        "@zhora_bot скинь мем",
        "@zhora_bot ахаха что это было",
        "@zhora_bot @vasya кто прав",
    ])
    def test_mentioned_question_or_request_is_recognized(self, text):
        """Questions and imperatives behind an @handle still read as requests."""
        assert looks_like_request(text) is True

    @pytest.mark.parametrize("text", [
        "@zhora_bot ок",
        "@zhora_bot ахаха",
        "@zhora_bot бля",
        "@zhora_bot сам такой",
        "@zhora_bot ну ты и фрукт",
    ])
    def test_mentioned_short_reaction_is_still_not_a_request(self, text):
        """Genuine short reactions keep their MEANINGLESS/BANTER verdict."""
        assert looks_like_request(text) is False

    def test_handle_does_not_count_toward_substantive_word_count(self):
        """A six-word message plus a handle is not promoted by word count alone."""
        assert looks_like_request("@zhora_bot ну вот опять началось это всё") is False

    async def test_mentioned_question_survives_a_meaningless_verdict(self):
        """The reported bug: a mentioned question no longer drops to an emoji."""
        node, _ = make_classify_node("MEANINGLESS")
        state = make_state(
            make_incoming(raw_text="@zhora_bot о чем я говорю"), should_respond=True
        )
        result = await node(state)
        assert result == {"filter_verdict": "MEANINGFUL"}


class TestClassify:
    async def test_meaningless_llm_response_returns_meaningless(self):
        node, _ = make_classify_node("MEANINGLESS")
        result = await node._ClassifyNode__classify("лол", FILTER_SYSTEM)
        assert result == "MEANINGLESS"

    async def test_meaningful_llm_response_returns_meaningful(self):
        node, _ = make_classify_node("MEANINGFUL")
        result = await node._ClassifyNode__classify("как дела?", FILTER_SYSTEM)
        assert result == "MEANINGFUL"

    async def test_llm_error_fails_open_as_meaningful(self):
        node, _ = make_classify_node(llm_error=Exception("API unavailable"))
        result = await node._ClassifyNode__classify("лол", FILTER_SYSTEM)
        assert result == "MEANINGFUL"

    async def test_lowercase_response_parsed_as_meaningless(self):
        node, _ = make_classify_node("meaningless")
        result = await node._ClassifyNode__classify("хаха", FILTER_SYSTEM)
        assert result == "MEANINGLESS"

    async def test_response_with_trailing_text_still_parsed(self):
        node, _ = make_classify_node("MEANINGLESS - just laughter")
        result = await node._ClassifyNode__classify("хаха", FILTER_SYSTEM)
        assert result == "MEANINGLESS"

    async def test_unrecognized_response_defaults_to_meaningful(self):
        node, _ = make_classify_node("UNKNOWN")
        result = await node._ClassifyNode__classify("что-то", FILTER_SYSTEM)
        assert result == "MEANINGFUL"

    async def test_meme_request_response_returns_meme_request(self):
        node, _ = make_classify_node("MEME_REQUEST")
        result = await node._ClassifyNode__classify("скинь мем", FILTER_SYSTEM)
        assert result == "MEME_REQUEST"

    async def test_group_profile_request_response_returns_group_profile_request(self):
        node, _ = make_classify_node("GROUP_PROFILE_REQUEST")
        result = await node._ClassifyNode__classify("оцени всем счастье", FILTER_SYSTEM)
        assert result == "GROUP_PROFILE_REQUEST"

    @pytest.mark.parametrize(
        "llm_response, expected",
        [
            ("MEANINGFUL", "MEANINGFUL"),
            ("MEANINGLESS", "MEANINGLESS"),
            ("BANTER", "BANTER"),
        ],
        ids=["meaningful", "meaningless", "banter"],
    )
    async def test_other_labels_are_not_swallowed_by_the_meme_check(self, llm_response, expected):
        """The MEME substring test must not capture the pre-existing labels."""
        node, _ = make_classify_node(llm_response)
        result = await node._ClassifyNode__classify("текст", FILTER_SYSTEM)
        assert result == expected


class TestGroupProfileMarkerRegex:
    @pytest.mark.parametrize(
        "text",
        [
            "оцени всем счастье от 1 до 10",
            "раздай всех роли из людей икс",
            "распредели каждому факультет",
            "rate everyone's happiness",
            "how does everybody rank",
            "assign a role to each of us",
        ],
    )
    def test_whole_chat_markers_match(self, text):
        assert GROUP_PROFILE_MARKER_RE.search(text)

    @pytest.mark.parametrize(
        "text",
        [
            "оцени Васю по шкале от 1 до 10",
            "кто из нас ты думаешь",
            "rate my happiness",
        ],
    )
    def test_single_person_requests_do_not_match(self, text):
        assert not GROUP_PROFILE_MARKER_RE.search(text)


class TestGroupProfileRequestDowngrade:
    async def test_no_whole_chat_marker_downgrades_to_meaningful(self):
        node, _ = make_classify_node("GROUP_PROFILE_REQUEST")
        text = "оцени Васю по шкале от 1 до 10"
        state = make_state(make_incoming(raw_text=text))
        result = await node._ClassifyNode__classify_addressed(state, text)
        assert result == "MEANINGFUL"

    async def test_whole_chat_marker_present_keeps_the_label(self):
        node, _ = make_classify_node("GROUP_PROFILE_REQUEST")
        text = "оцени всем счастье от 1 до 10"
        state = make_state(make_incoming(raw_text=text))
        result = await node._ClassifyNode__classify_addressed(state, text)
        assert result == "GROUP_PROFILE_REQUEST"


class TestParseLabel:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("MEME", "MEME_REQUEST"),
            ("profile", "GROUP_PROFILE_REQUEST"),
            ("NOT_ADDRESSED", "NOT_ADDRESSED"),
            ("insult", "BOT_INSULT"),
            ("banter", "BANTER"),
            ("meaningless", "MEANINGLESS"),
            ("???", "MEANINGFUL"),
        ],
    )
    def test_maps_free_text_to_label(self, content, expected):
        assert parse_label(content) == expected
