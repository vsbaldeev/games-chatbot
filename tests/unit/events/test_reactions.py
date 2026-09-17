"""Emoji-reaction tracking: Telegram sends the full old/new reaction sets on
every update, not a single delta, so the delta is computed here."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.events.reactions import diff_reactions, handle_message_reaction, reaction_key


def make_emoji_reaction(emoji: str) -> MagicMock:
    reaction = MagicMock()
    reaction.type = "emoji"
    reaction.emoji = emoji
    return reaction


def make_custom_reaction(custom_emoji_id: str) -> MagicMock:
    reaction = MagicMock()
    reaction.type = "custom_emoji"
    reaction.custom_emoji_id = custom_emoji_id
    return reaction


def make_paid_reaction() -> MagicMock:
    reaction = MagicMock()
    reaction.type = "paid"
    return reaction


class TestReactionKey:
    def test_emoji_reaction_key_is_the_emoji(self):
        assert reaction_key(make_emoji_reaction("🤡")) == "🤡"

    def test_custom_emoji_reaction_key_is_prefixed(self):
        assert reaction_key(make_custom_reaction("12345")) == "custom:12345"

    def test_paid_reaction_key_is_constant(self):
        assert reaction_key(make_paid_reaction()) == "paid"


class TestDiffReactions:
    def test_new_reaction_is_added(self):
        added, removed = diff_reactions([], [make_emoji_reaction("🤡")])
        assert added == ["🤡"]
        assert removed == []

    def test_removed_reaction_is_removed(self):
        added, removed = diff_reactions([make_emoji_reaction("🤡")], [])
        assert added == []
        assert removed == ["🤡"]

    def test_unchanged_reaction_is_neither(self):
        added, removed = diff_reactions([make_emoji_reaction("🤡")], [make_emoji_reaction("🤡")])
        assert added == []
        assert removed == []

    def test_swap_is_one_added_one_removed(self):
        added, removed = diff_reactions([make_emoji_reaction("🤡")], [make_emoji_reaction("😂")])
        assert added == ["😂"]
        assert removed == ["🤡"]


class TestHandleMessageReaction:
    async def test_ignores_non_group_chats(self):
        update = MagicMock()
        update.effective_chat.type = "private"
        with patch("src.events.reactions.message_feedback.add_reaction", AsyncMock()) as add_reaction:
            await handle_message_reaction(update, MagicMock())
        add_reaction.assert_not_awaited()

    async def test_applies_added_and_removed_reactions(self):
        update = MagicMock()
        update.effective_chat.type = "supergroup"
        update.message_reaction.chat.id = 1000
        update.message_reaction.message_id = 55
        update.message_reaction.old_reaction = [make_emoji_reaction("🤡")]
        update.message_reaction.new_reaction = [make_emoji_reaction("😂")]
        with patch("src.events.reactions.message_feedback.add_reaction", AsyncMock()) as add_reaction, \
             patch("src.events.reactions.message_feedback.remove_reaction", AsyncMock()) as remove_reaction:
            await handle_message_reaction(update, MagicMock())
        add_reaction.assert_awaited_once_with(chat_id=1000, message_id=55, emoji="😂")
        remove_reaction.assert_awaited_once_with(chat_id=1000, message_id=55, emoji="🤡")
