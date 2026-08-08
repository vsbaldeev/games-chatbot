"""Life-post writer tests — build_engagement_lines/build_episode_prompt.

Life posts never mention chat members (2026-08-07 decision, see
docs/superpowers/specs/2026-08-07-reduce-bot-absurdity-design.md) — the
engagement instruction is always the same closing-question prompt, with no
member-mention mode to select between.
"""

from src.life.writer import build_engagement_lines, build_episode_prompt


class TestBuildEngagementLines:
    def test_always_returns_the_chat_question_instruction(self):
        lines = build_engagement_lines()
        joined = "\n".join(lines)
        assert "вопросом" in joined

    def test_never_mentions_a_member(self):
        lines = build_engagement_lines()
        joined = "\n".join(lines)
        assert "@" not in joined


class TestBuildEpisodePromptNoMemberParams:
    def test_builds_without_mode_or_mention_arguments(self):
        prompt = build_episode_prompt(
            recent_episodes=[], facts=[], recent_activities=[], post_format="story",
        )
        assert "Формат этого поста: story." in prompt
        assert "@" not in prompt
