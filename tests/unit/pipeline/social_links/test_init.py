"""Tests for the social_links package's own shared helpers."""

from src.pipeline.social_links import render_comment_lines


class TestRenderCommentLines:
    def test_renders_text_and_like_count(self):
        comments = [{"text": "nice", "like_count": 5}]
        result = render_comment_lines(comments, max_comments=10, char_limit=200)
        assert result == "[Топ-комментарии]:\n- (5 лайков) nice"

    def test_truncates_long_comments(self):
        comments = [{"text": "a" * 300, "like_count": 1}]
        result = render_comment_lines(comments, max_comments=10, char_limit=200)
        assert "a" * 200 + "…" in result
        assert "a" * 201 not in result

    def test_caps_at_max_comments(self):
        comments = [{"text": f"comment {i}", "like_count": i} for i in range(20)]
        result = render_comment_lines(comments, max_comments=3, char_limit=200)
        assert result.count("лайков") == 3

    def test_empty_comments_returns_empty_string(self):
        assert render_comment_lines([], max_comments=10, char_limit=200) == ""

    def test_blank_text_comments_are_skipped(self):
        comments = [{"text": "   ", "like_count": 5}, {"text": "real", "like_count": 1}]
        result = render_comment_lines(comments, max_comments=10, char_limit=200)
        assert "real" in result
        assert result.count("лайков") == 1
