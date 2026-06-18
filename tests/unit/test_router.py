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
        incoming = make_incoming(raw_text="обычное сообщение без упоминания бота")
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
