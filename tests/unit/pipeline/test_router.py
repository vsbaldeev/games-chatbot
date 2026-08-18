"""
Router routing-decision tests.

Exercises MessageRouter.__decide() directly — the pure function that picks
should_respond + response_trigger from the incoming message. DB calls the
addressing gate makes are mocked via mock_link_repost_lookup below; no LLM
calls happen here; the storage side of __call__ is not under test.

Scenarios anchored to real bugs:
  e8fa36c — forwarded posts must never trigger a response
  c21fe1c — forwarded messages must not be routed anywhere active
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline.router import INSTAGRAM_REEL_DAILY_CAP_REPLIES, MessageRouter
from src.pipeline.social_links.instagram_reel import INSTAGRAM_REEL_DAILY_CAP
from tests.builders import make_incoming, make_telegram_message

BOT_USERNAME = "testbot"
BOT_ID = 123456789  # matches TELEGRAM_TOKEN prefix set in conftest


@pytest.fixture
def router() -> MessageRouter:
    return MessageRouter(bot_username=BOT_USERNAME, bot_id=BOT_ID)


@pytest.fixture(autouse=True)
def mock_link_repost_lookup():
    """Default every test to 'not a link-repost row' — today's blanket rule.

    Individual tests override ``return_value`` to simulate a reply landing
    on the bot's link-repost message and exercise the addressing gate.
    """
    with patch(
        "src.pipeline.router.unified_messages.get_by_id",
        new_callable=AsyncMock, return_value=None,
    ) as mock:
        yield mock


async def call_decide(router: MessageRouter, incoming: dict) -> tuple[bool, str]:
    """Invoke the private routing decision method, returning its two headline fields."""
    telegram_message = incoming["update"].message
    update = await router._MessageRouter__decide(incoming, telegram_message)
    return update["should_respond"], update["response_trigger"]


async def call_decide_full(router: MessageRouter, incoming: dict) -> dict:
    """Invoke the private routing decision method, returning the whole state update."""
    telegram_message = incoming["update"].message
    return await router._MessageRouter__decide(incoming, telegram_message)


class TestForwardedMessages:
    """Forwarded messages must never produce a response (e8fa36c, c21fe1c)."""

    async def test_forwarded_plain_text_does_not_respond(self, router):
        incoming = make_incoming(is_forwarded=True, raw_text="check this out")
        should_respond, _ = await call_decide(router, incoming)
        assert not should_respond

    async def test_forwarded_text_with_bot_mention_does_not_respond(self, router):
        incoming = make_incoming(
            is_forwarded=True,
            raw_text=f"@{BOT_USERNAME} что думаешь?",
        )
        should_respond, _ = await call_decide(router, incoming)
        assert not should_respond

    async def test_forwarded_voice_does_not_respond(self, router):
        incoming = make_incoming(is_forwarded=True, media_type="voice")
        should_respond, _ = await call_decide(router, incoming)
        assert not should_respond

    async def test_forwarded_photo_does_not_respond(self, router):
        incoming = make_incoming(is_forwarded=True, media_type="photo")
        should_respond, _ = await call_decide(router, incoming)
        assert not should_respond


class TestExplicitMention:
    """@mention in text body triggers an explicit response."""

    async def test_mention_in_text_responds_explicitly(self, router):
        incoming = make_incoming(raw_text=f"@{BOT_USERNAME} как дела?")
        should_respond, trigger = await call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    async def test_mention_matching_is_case_insensitive(self, router):
        incoming = make_incoming(raw_text=f"@{BOT_USERNAME.upper()} привет")
        should_respond, trigger = await call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    async def test_plain_text_without_mention_does_not_respond(self, router):
        incoming = make_incoming(raw_text="обычное сообщение ни о чём")
        should_respond, _ = await call_decide(router, incoming)
        assert not should_respond


class TestReplyToBotMessage:
    """Reply to a bot message is treated as an explicit trigger (core reply-chain flow)."""

    async def test_reply_to_bot_responds_explicitly(self, router):
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID)
        incoming = make_incoming(telegram_message=telegram_message)
        should_respond, trigger = await call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    async def test_reply_to_bot_without_mention_still_explicit(self, router):
        """A reply to a bot message should respond even without @mention text."""
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID)
        incoming = make_incoming(raw_text="ок понял", telegram_message=telegram_message)
        should_respond, trigger = await call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    async def test_reply_to_other_user_does_not_respond(self, router):
        telegram_message = make_telegram_message(reply_to_user_id=99999)
        incoming = make_incoming(telegram_message=telegram_message)
        should_respond, _ = await call_decide(router, incoming)
        assert not should_respond


class TestMediaMessages:
    async def test_voice_with_bot_mention_in_caption_responds_explicitly(self, router):
        telegram_message = make_telegram_message(caption=f"@{BOT_USERNAME}")
        incoming = make_incoming(media_type="voice", telegram_message=telegram_message)
        should_respond, trigger = await call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    async def test_voice_reply_to_bot_responds_explicitly(self, router):
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID)
        incoming = make_incoming(media_type="voice", telegram_message=telegram_message)
        should_respond, trigger = await call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    async def test_sticker_never_responds(self, router):
        incoming = make_incoming(media_type="sticker")
        should_respond, _ = await call_decide(router, incoming)
        assert not should_respond

    async def test_unaddressed_photo_never_responds(self, router):
        incoming = make_incoming(media_type="photo")
        should_respond, trigger = await call_decide(router, incoming)
        assert not should_respond
        assert trigger == "random"

    async def test_unaddressed_voice_never_responds(self, router):
        incoming = make_incoming(media_type="voice")
        should_respond, trigger = await call_decide(router, incoming)
        assert not should_respond
        assert trigger == "random"

    async def test_unaddressed_video_note_never_responds(self, router):
        incoming = make_incoming(media_type="video_note")
        should_respond, trigger = await call_decide(router, incoming)
        assert not should_respond
        assert trigger == "random"

    async def test_unaddressed_video_never_responds(self, router):
        incoming = make_incoming(media_type="video")
        should_respond, trigger = await call_decide(router, incoming)
        assert not should_respond
        assert trigger == "random"


class TestSocialLinkDetection:
    """Instagram/YouTube link auto-detection (mirrors Shorts detection).

    Each test uses a distinct link id so the handlers' module-level TtlGate
    singletons (shared across the whole test session, same as shorts.dedup_gate)
    never collide between tests.
    """

    def test_no_link_returns_none(self, router):
        msg = make_incoming(raw_text="just chatting, no links here")
        assert router._MessageRouter__detect_social_link(msg) is None

    def test_instagram_reel_link_triggers_social_link(self, router):
        msg = make_incoming(raw_text="https://www.instagram.com/reel/RouterTest01/")
        result = router._MessageRouter__detect_social_link(msg)
        # social_link_cap_remaining depends on how many prior Instagram hits
        # this session's shared daily_cap_gate has already recorded for
        # chat_id=1000 — asserted precisely (from a clean chat) in
        # TestSocialLinkDailyCap below; here just check the shape.
        cap_remaining = result.pop("social_link_cap_remaining")
        assert isinstance(cap_remaining, int)
        assert result == {
            "should_respond": True,
            "response_trigger": "social_link",
            "social_link_handler": "instagram_reel",
            "social_link_url": "https://www.instagram.com/reel/RouterTest01/",
            "link_message_is_bare": True,
        }

    def test_youtube_watch_link_triggers_social_link(self, router):
        msg = make_incoming(raw_text="https://www.youtube.com/watch?v=RouterTest03")
        result = router._MessageRouter__detect_social_link(msg)
        assert result["social_link_handler"] == "youtube_video"

    def test_repost_within_dedup_window_is_ignored(self, router):
        url = "https://www.instagram.com/reel/RouterTest04/"
        msg = make_incoming(raw_text=url)
        assert router._MessageRouter__detect_social_link(msg) is not None
        assert router._MessageRouter__detect_social_link(msg) is None

    def test_instagram_priority_over_youtube_in_same_message(self, router):
        msg = make_incoming(
            raw_text=(
                "https://www.instagram.com/reel/RouterTest05/ and also "
                "https://www.youtube.com/watch?v=RouterTest05b"
            )
        )
        result = router._MessageRouter__detect_social_link(msg)
        assert result["social_link_handler"] == "instagram_reel"

    def test_gate_rejected_leader_does_not_fall_through_to_youtube(self, router):
        """First-match commitment holds even when the leader's own gate rejects it.

        Instagram's regex matches first in the combined message. Once its
        dedup gate has already fired for this exact Reel, the whole message
        must be dropped — the YouTube link later in the same text is never
        considered, even though it would pass its own gate cleanly.
        """
        reel_url = "https://www.instagram.com/reel/RouterTest06/"
        router._MessageRouter__detect_social_link(make_incoming(raw_text=reel_url))

        combined_msg = make_incoming(
            raw_text=(
                f"{reel_url} and also "
                "https://www.youtube.com/watch?v=RouterTest06b"
            )
        )
        assert router._MessageRouter__detect_social_link(combined_msg) is None


class TestSocialLinkDailyCap:
    """Instagram alone is throttled; YouTube's handler has no cap at all.

    Distinct chat ids per test, same reason as TestSocialLinkDetection: the
    handlers' daily_cap_gate TtlGate singletons are module-level and shared
    across the whole test session.
    """

    def test_instagram_cap_exceeded_replies_instead_of_silence(self, router):
        chat_id = 90001
        for index in range(INSTAGRAM_REEL_DAILY_CAP):
            msg = make_incoming(
                chat_id=chat_id,
                raw_text=f"https://www.instagram.com/reel/DailyCapA{index:03d}/",
            )
            assert router._MessageRouter__detect_social_link(msg) is not None

        over_cap_msg = make_incoming(
            chat_id=chat_id, raw_text="https://www.instagram.com/reel/DailyCapOver/",
        )
        result = router._MessageRouter__detect_social_link(over_cap_msg)
        assert result["should_respond"] is False
        assert result["response"] in INSTAGRAM_REEL_DAILY_CAP_REPLIES

    def test_cap_remaining_counts_down_from_a_clean_chat(self, router):
        chat_id = 90003
        first = router._MessageRouter__detect_social_link(
            make_incoming(chat_id=chat_id, raw_text="https://www.instagram.com/reel/DailyCapB000/")
        )
        assert first["social_link_cap_remaining"] == INSTAGRAM_REEL_DAILY_CAP - 1

        second = router._MessageRouter__detect_social_link(
            make_incoming(chat_id=chat_id, raw_text="https://www.instagram.com/reel/DailyCapB001/")
        )
        assert second["social_link_cap_remaining"] == INSTAGRAM_REEL_DAILY_CAP - 2

    def test_youtube_video_handler_is_never_capped(self, router):
        chat_id = 90002
        for index in range(INSTAGRAM_REEL_DAILY_CAP + 5):
            msg = make_incoming(
                chat_id=chat_id,
                raw_text=f"https://www.youtube.com/watch?v=DailyCapY{index:03d}",
            )
            result = router._MessageRouter__detect_social_link(msg)
            assert result["social_link_handler"] == "youtube_video"
            assert result["should_respond"] is True
            assert result["social_link_cap_remaining"] is None


class TestLinkMessageIsBare:
    """Only a message that is nothing but a link may later be deleted.

    Distinct link ids per test, same reason as TestSocialLinkDetection: the
    dedup gates are module-level singletons shared across the session.
    """

    def test_bare_shorts_link_is_marked_bare(self, router):
        msg = make_incoming(raw_text="https://www.youtube.com/shorts/RouterBare01")
        result = router._MessageRouter__detect_shorts(msg)
        assert result["response_trigger"] == "youtube_short"
        assert result["link_message_is_bare"] is True

    def test_shorts_link_with_user_text_is_not_bare(self, router):
        msg = make_incoming(
            raw_text="гляньте какая дичь https://www.youtube.com/shorts/RouterBare02"
        )
        result = router._MessageRouter__detect_shorts(msg)
        assert result["link_message_is_bare"] is False

    def test_shorts_link_with_tracking_params_is_bare(self, router):
        msg = make_incoming(raw_text="https://youtube.com/shorts/RouterBare03?si=xYz")
        result = router._MessageRouter__detect_shorts(msg)
        assert result["link_message_is_bare"] is True

    def test_bare_youtube_watch_link_is_marked_bare(self, router):
        msg = make_incoming(raw_text="https://www.youtube.com/watch?v=RouterBare04")
        result = router._MessageRouter__detect_social_link(msg)
        assert result["link_message_is_bare"] is True

    def test_instagram_link_with_user_text_is_not_bare(self, router):
        msg = make_incoming(
            raw_text="смотри https://www.instagram.com/reel/RouterBare05/"
        )
        result = router._MessageRouter__detect_social_link(msg)
        assert result["link_message_is_bare"] is False


class TestLinkRepostReplyGate:
    """A reply to the bot's link-repost message needs a mention or a
    request-like phrasing to count as addressing the bot (see the design
    spec's Addressing Gate). Any other bot message keeps the blanket rule —
    mock_link_repost_lookup defaults to None, i.e. 'not a link-repost row'.
    """

    async def test_bare_reply_to_link_repost_message_does_not_respond(
        self, router, mock_link_repost_lookup
    ):
        mock_link_repost_lookup.return_value = {"link_material": "материал о видео"}
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID, text="ору")
        incoming = make_incoming(raw_text="ору", telegram_message=telegram_message)
        should_respond, trigger = await call_decide(router, incoming)
        assert not should_respond
        assert trigger == "random"

    async def test_question_reply_to_link_repost_message_responds_explicitly(
        self, router, mock_link_repost_lookup
    ):
        mock_link_repost_lookup.return_value = {"link_material": "материал о видео"}
        question = "а что он в конце сказал?"
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID, text=question)
        incoming = make_incoming(raw_text=question, telegram_message=telegram_message)
        should_respond, trigger = await call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    async def test_mentioning_reply_to_link_repost_message_responds_explicitly(
        self, router, mock_link_repost_lookup
    ):
        mock_link_repost_lookup.return_value = {"link_material": "материал о видео"}
        mentioning_text = f"@{BOT_USERNAME} ору"
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID, text=mentioning_text)
        incoming = make_incoming(raw_text=mentioning_text, telegram_message=telegram_message)
        should_respond, trigger = await call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    async def test_reply_to_non_link_repost_bot_message_keeps_blanket_rule(
        self, router, mock_link_repost_lookup
    ):
        # mock_link_repost_lookup already defaults to None (an ordinary bot
        # message, e.g. a joke) — also covers a purged/missing row: never
        # gate more aggressively on missing data than on a confirmed
        # non-link-repost row.
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID)
        incoming = make_incoming(raw_text="ору", telegram_message=telegram_message)
        should_respond, trigger = await call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    async def test_gate_requires_mention_for_non_text_replies(
        self, router, mock_link_repost_lookup
    ):
        # The router runs before transcription, so a real voice message's
        # telegram_message.text/.caption are both None at this point (no
        # caption on the voice note) — __decide derives its addressing text
        # from the Telegram object, not msg["raw_text"], so
        # make_telegram_message() with neither text= nor caption= set
        # (its defaults) genuinely reproduces "no text yet" — looks_like_request
        # has nothing to judge, and only an explicit mention can satisfy the
        # gate.
        mock_link_repost_lookup.return_value = {"link_material": "материал о видео"}
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID)
        incoming = make_incoming(media_type="voice", telegram_message=telegram_message)
        should_respond, trigger = await call_decide(router, incoming)
        assert not should_respond
        assert trigger == "random"

    async def test_lookup_is_only_called_when_replying_to_the_bot(
        self, router, mock_link_repost_lookup
    ):
        incoming = make_incoming(raw_text="обычное сообщение ни о чём")
        await call_decide(router, incoming)
        mock_link_repost_lookup.assert_not_called()


class TestBroadcastReplyFlag:
    """A reply to a group-wide announcement is flagged so the filter may judge
    whether it addresses the bot at all (2026-08-18 addressee gate). The flag
    only marks the message — the router still routes it as 'explicit'.
    """

    async def test_bare_reply_to_broadcast_is_flagged(
        self, router, mock_link_repost_lookup
    ):
        mock_link_repost_lookup.return_value = {"is_broadcast": True}
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID, text="Гениально)")
        incoming = make_incoming(raw_text="Гениально)", telegram_message=telegram_message)
        update = await call_decide_full(router, incoming)
        assert update["broadcast_reply"] is True
        assert update["should_respond"] is True
        assert update["response_trigger"] == "explicit"

    async def test_mentioning_reply_to_broadcast_is_not_flagged(
        self, router, mock_link_repost_lookup
    ):
        """An explicit @mention is unambiguous addressing — never second-guess it."""
        mock_link_repost_lookup.return_value = {"is_broadcast": True}
        text = f"@{BOT_USERNAME} почему мне 2 из 10?"
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID, text=text)
        incoming = make_incoming(raw_text=text, telegram_message=telegram_message)
        update = await call_decide_full(router, incoming)
        assert update.get("broadcast_reply", False) is False

    async def test_reply_to_ordinary_bot_message_is_not_flagged(
        self, router, mock_link_repost_lookup
    ):
        mock_link_repost_lookup.return_value = {"is_broadcast": False}
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID, text="ору")
        incoming = make_incoming(raw_text="ору", telegram_message=telegram_message)
        update = await call_decide_full(router, incoming)
        assert update.get("broadcast_reply", False) is False

    async def test_missing_row_is_not_flagged(self, router, mock_link_repost_lookup):
        """A purged row must never gate more aggressively than a confirmed one."""
        mock_link_repost_lookup.return_value = None
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID, text="ору")
        incoming = make_incoming(raw_text="ору", telegram_message=telegram_message)
        update = await call_decide_full(router, incoming)
        assert update.get("broadcast_reply", False) is False

    async def test_broadcast_lookup_happens_once(self, router, mock_link_repost_lookup):
        """One row serves both the link-repost gate and the broadcast flag."""
        mock_link_repost_lookup.return_value = {"is_broadcast": True}
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID, text="Гениально)")
        incoming = make_incoming(raw_text="Гениально)", telegram_message=telegram_message)
        await call_decide_full(router, incoming)
        assert mock_link_repost_lookup.await_count == 1
