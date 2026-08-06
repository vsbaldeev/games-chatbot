"""
Router routing-decision tests.

Exercises MessageRouter.__decide() directly — the pure function that picks
should_respond + response_trigger from the incoming message.  No DB or LLM
calls happen here; the storage side of __call__ is not under test.

Scenarios anchored to real bugs:
  e8fa36c — forwarded posts must never trigger a response
  c21fe1c — forwarded messages must not be routed anywhere active
"""

import pytest

from src.pipeline.router import MessageRouter
from tests.builders import make_incoming, make_telegram_message

BOT_USERNAME = "testbot"
BOT_ID = 123456789  # matches TELEGRAM_TOKEN prefix set in conftest


@pytest.fixture
def router() -> MessageRouter:
    return MessageRouter(bot_username=BOT_USERNAME, bot_id=BOT_ID)


def call_decide(router: MessageRouter, incoming: dict) -> tuple[bool, str]:
    """Invoke the private routing decision method."""
    telegram_message = incoming["update"].message
    return router._MessageRouter__decide(incoming, telegram_message)


class TestForwardedMessages:
    """Forwarded messages must never produce a response (e8fa36c, c21fe1c)."""

    def test_forwarded_plain_text_does_not_respond(self, router):
        incoming = make_incoming(is_forwarded=True, raw_text="check this out")
        should_respond, _ = call_decide(router, incoming)
        assert not should_respond

    def test_forwarded_text_with_bot_mention_does_not_respond(self, router):
        incoming = make_incoming(
            is_forwarded=True,
            raw_text=f"@{BOT_USERNAME} что думаешь?",
        )
        should_respond, _ = call_decide(router, incoming)
        assert not should_respond

    def test_forwarded_voice_does_not_respond(self, router):
        incoming = make_incoming(is_forwarded=True, media_type="voice")
        should_respond, _ = call_decide(router, incoming)
        assert not should_respond

    def test_forwarded_photo_does_not_respond(self, router):
        incoming = make_incoming(is_forwarded=True, media_type="photo")
        should_respond, _ = call_decide(router, incoming)
        assert not should_respond


class TestExplicitMention:
    """@mention in text body triggers an explicit response."""

    def test_mention_in_text_responds_explicitly(self, router):
        incoming = make_incoming(raw_text=f"@{BOT_USERNAME} как дела?")
        should_respond, trigger = call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    def test_mention_matching_is_case_insensitive(self, router):
        incoming = make_incoming(raw_text=f"@{BOT_USERNAME.upper()} привет")
        should_respond, trigger = call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    def test_plain_text_without_mention_does_not_respond(self, router):
        incoming = make_incoming(raw_text="обычное сообщение ни о чём")
        should_respond, _ = call_decide(router, incoming)
        assert not should_respond


class TestReplyToBotMessage:
    """Reply to a bot message is treated as an explicit trigger (core reply-chain flow)."""

    def test_reply_to_bot_responds_explicitly(self, router):
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID)
        incoming = make_incoming(telegram_message=telegram_message)
        should_respond, trigger = call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    def test_reply_to_bot_without_mention_still_explicit(self, router):
        """A reply to a bot message should respond even without @mention text."""
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID)
        incoming = make_incoming(raw_text="ок понял", telegram_message=telegram_message)
        should_respond, trigger = call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    def test_reply_to_other_user_does_not_respond(self, router):
        telegram_message = make_telegram_message(reply_to_user_id=99999)
        incoming = make_incoming(telegram_message=telegram_message)
        should_respond, _ = call_decide(router, incoming)
        assert not should_respond


class TestMediaMessages:
    def test_voice_with_bot_mention_in_caption_responds_explicitly(self, router):
        telegram_message = make_telegram_message(caption=f"@{BOT_USERNAME}")
        incoming = make_incoming(media_type="voice", telegram_message=telegram_message)
        should_respond, trigger = call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    def test_voice_reply_to_bot_responds_explicitly(self, router):
        telegram_message = make_telegram_message(reply_to_user_id=BOT_ID)
        incoming = make_incoming(media_type="voice", telegram_message=telegram_message)
        should_respond, trigger = call_decide(router, incoming)
        assert should_respond
        assert trigger == "explicit"

    def test_sticker_never_responds(self, router):
        incoming = make_incoming(media_type="sticker")
        should_respond, _ = call_decide(router, incoming)
        assert not should_respond


class TestSocialLinkDetection:
    """Instagram/Reddit/YouTube link auto-detection (mirrors Shorts detection).

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
        assert result == {
            "should_respond": True,
            "response_trigger": "social_link",
            "social_link_handler": "instagram_reel",
            "social_link_url": "https://www.instagram.com/reel/RouterTest01/",
        }

    def test_reddit_link_triggers_social_link(self, router):
        msg = make_incoming(
            raw_text="https://www.reddit.com/r/funny/comments/routertest02/some_title/"
        )
        result = router._MessageRouter__detect_social_link(msg)
        assert result["social_link_handler"] == "reddit_post"
        assert result["response_trigger"] == "social_link"

    def test_youtube_watch_link_triggers_social_link(self, router):
        msg = make_incoming(raw_text="https://www.youtube.com/watch?v=RouterTest03")
        result = router._MessageRouter__detect_social_link(msg)
        assert result["social_link_handler"] == "youtube_video"

    def test_repost_within_dedup_window_is_ignored(self, router):
        url = "https://www.instagram.com/reel/RouterTest04/"
        msg = make_incoming(raw_text=url)
        assert router._MessageRouter__detect_social_link(msg) is not None
        assert router._MessageRouter__detect_social_link(msg) is None

    def test_instagram_priority_over_reddit_in_same_message(self, router):
        msg = make_incoming(
            raw_text=(
                "https://www.instagram.com/reel/RouterTest05/ and also "
                "https://www.reddit.com/r/funny/comments/routertest05b/title/"
            )
        )
        result = router._MessageRouter__detect_social_link(msg)
        assert result["social_link_handler"] == "instagram_reel"

    def test_gate_rejected_leader_does_not_fall_through_to_reddit(self, router):
        """First-match commitment holds even when the leader's own gate rejects it.

        Instagram's regex matches first in the combined message. Once its
        dedup gate has already fired for this exact Reel, the whole message
        must be dropped — the Reddit link later in the same text is never
        considered, even though it would pass its own gate cleanly.
        """
        reel_url = "https://www.instagram.com/reel/RouterTest06/"
        router._MessageRouter__detect_social_link(make_incoming(raw_text=reel_url))

        combined_msg = make_incoming(
            raw_text=(
                f"{reel_url} and also "
                "https://www.reddit.com/r/funny/comments/routertest06b/title/"
            )
        )
        assert router._MessageRouter__detect_social_link(combined_msg) is None
