"""RedditPostHandler tests — link detection and JSON-API fetch/compose."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.pipeline.social_links import SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT
from src.pipeline.social_links.reddit_post import RedditPostHandler

HTTPX_CLIENT_PATCH_TARGET = "src.pipeline.social_links.reddit_post.httpx.AsyncClient"


@pytest.fixture
def handler() -> RedditPostHandler:
    return RedditPostHandler()


class TestExtract:
    def test_no_link_returns_none(self, handler):
        assert handler.extract("just some text, no links") is None

    def test_comments_url_extracts_id_and_canonical_url(self, handler):
        text = "check this https://www.reddit.com/r/funny/comments/1abcxyz/some_title/"
        item_id, url = handler.extract(text)
        assert item_id == "1abcxyz"
        assert url == "https://www.reddit.com/comments/1abcxyz"

    def test_short_link_extracts_id(self, handler):
        item_id, url = handler.extract("look at https://redd.it/1abcxyz here")
        assert item_id == "1abcxyz"
        assert url == "https://www.reddit.com/comments/1abcxyz"

    def test_share_link_with_tracking_query_string_extracts_clean_canonical_url(self, handler):
        text = (
            "look https://www.reddit.com/r/funny/comments/1abcxyz/some_title/"
            "?utm_source=share&utm_medium=web3x"
        )
        item_id, url = handler.extract(text)
        assert item_id == "1abcxyz"
        assert url == "https://www.reddit.com/comments/1abcxyz"
        assert "utm_source" not in url

    def test_old_reddit_subdomain_is_matched(self, handler):
        text = "https://old.reddit.com/r/funny/comments/1abcxyz/some_title/"
        item_id, url = handler.extract(text)
        assert item_id == "1abcxyz"
        assert url == "https://www.reddit.com/comments/1abcxyz"

    def test_mobile_reddit_subdomain_is_matched(self, handler):
        text = "https://m.reddit.com/r/funny/comments/1abcxyz/some_title/"
        item_id, url = handler.extract(text)
        assert item_id == "1abcxyz"
        assert url == "https://www.reddit.com/comments/1abcxyz"

    def test_scheme_less_link_is_matched(self, handler):
        text = "check reddit.com/r/funny/comments/1abcxyz/some_title/ out"
        item_id, url = handler.extract(text)
        assert item_id == "1abcxyz"
        assert url == "https://www.reddit.com/comments/1abcxyz"


def make_reddit_payload(title="Some title", subreddit="funny", selftext="", comments=None):
    comments = comments or []
    return [
        {"data": {"children": [{"data": {"title": title, "subreddit": subreddit, "selftext": selftext}}]}},
        {"data": {"children": [{"data": comment} for comment in comments]}},
    ]


def make_mock_httpx_client(payload=None, raise_error: Exception | None = None) -> MagicMock:
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    if raise_error is not None:
        mock_response.raise_for_status.side_effect = raise_error
    else:
        mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)
    return mock_client


class TestFetch:
    async def test_successful_fetch_composes_header_selftext_and_comments(self, handler):
        payload = make_reddit_payload(
            title="Fix a fence",
            subreddit="therewasanattempt",
            selftext="lol look at this",
            comments=[{"author": "alice", "body": "nice job", "score": 42}],
        )
        mock_client = make_mock_httpx_client(payload=payload)
        with patch(HTTPX_CLIENT_PATCH_TARGET, return_value=mock_client):
            result = await handler.fetch(
                "https://www.reddit.com/r/therewasanattempt/comments/1abcxyz/fix/"
            )
        assert result is not None
        assert result["video_bytes"] is None
        assert "therewasanattempt" in result["content_block"]
        assert "Fix a fence" in result["content_block"]
        assert "lol look at this" in result["content_block"]
        assert "nice job" in result["content_block"]
        assert "(42 очков)" in result["content_block"]

    async def test_comments_sorted_by_score_descending(self, handler):
        payload = make_reddit_payload(comments=[
            {"author": "low", "body": "low score comment", "score": 1},
            {"author": "high", "body": "high score comment", "score": 999},
        ])
        mock_client = make_mock_httpx_client(payload=payload)
        with patch(HTTPX_CLIENT_PATCH_TARGET, return_value=mock_client):
            result = await handler.fetch("https://www.reddit.com/r/x/comments/abc/y/")
        content = result["content_block"]
        assert content.index("high score comment") < content.index("low score comment")

    async def test_deleted_and_automoderator_comments_filtered(self, handler):
        payload = make_reddit_payload(comments=[
            {"author": "AutoModerator", "body": "rules apply", "score": 500},
            {"author": "someone", "body": "[deleted]", "score": 100},
            {"author": "real_user", "body": "actual comment", "score": 5},
        ])
        mock_client = make_mock_httpx_client(payload=payload)
        with patch(HTTPX_CLIENT_PATCH_TARGET, return_value=mock_client):
            result = await handler.fetch("https://www.reddit.com/r/x/comments/abc/y/")
        assert "rules apply" not in result["content_block"]
        assert "actual comment" in result["content_block"]

    async def test_http_error_returns_none(self, handler):
        mock_client = make_mock_httpx_client(raise_error=httpx.ConnectError("boom"))
        with patch(HTTPX_CLIENT_PATCH_TARGET, return_value=mock_client):
            result = await handler.fetch("https://www.reddit.com/r/x/comments/abc/y/")
        assert result is None

    async def test_missing_title_returns_none(self, handler):
        payload = make_reddit_payload(title="")
        mock_client = make_mock_httpx_client(payload=payload)
        with patch(HTTPX_CLIENT_PATCH_TARGET, return_value=mock_client):
            result = await handler.fetch("https://www.reddit.com/r/x/comments/abc/y/")
        assert result is None

    async def test_selftext_at_limit_is_untouched(self, handler):
        selftext = "a" * SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT
        payload = make_reddit_payload(selftext=selftext)
        mock_client = make_mock_httpx_client(payload=payload)
        with patch(HTTPX_CLIENT_PATCH_TARGET, return_value=mock_client):
            result = await handler.fetch("https://www.reddit.com/r/x/comments/abc/y/")
        assert selftext in result["content_block"]
        assert "…" not in result["content_block"]

    async def test_selftext_over_limit_is_truncated(self, handler):
        long_selftext = "a" * (SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT + 1000)
        payload = make_reddit_payload(selftext=long_selftext)
        mock_client = make_mock_httpx_client(payload=payload)
        with patch(HTTPX_CLIENT_PATCH_TARGET, return_value=mock_client):
            result = await handler.fetch("https://www.reddit.com/r/x/comments/abc/y/")
        expected = "a" * SOCIAL_LINK_DESCRIPTION_CHAR_LIMIT + "…"
        assert expected in result["content_block"]
        assert long_selftext not in result["content_block"]
