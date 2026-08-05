"""Unit tests for the 9gag meme source.

Covers two units in isolation:
  - parse_ninegag: pure JSON-to-candidate filtering logic
  - fetch: async HTTP path with the httpx client fully mocked
"""

from unittest.mock import AsyncMock, MagicMock

import httpx

from src.memes.sources import ninegag
from src.memes.sources.base import MemeCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_post(
    *,
    post_id: str = "aByApLA",
    title: str = "Funny title",
    post_type: str = "Photo",
    nsfw: int = 0,
    image_url: str | None = "https://img-9gag-fun.9cache.com/photo/aByApLA_700b.jpg",
) -> dict:
    """Build a single 9gag post dict shaped like the real feed payload."""
    images = {"image700": {"url": image_url}} if image_url is not None else {}
    return {
        "id": post_id,
        "title": title,
        "type": post_type,
        "nsfw": nsfw,
        "images": images,
    }


def make_payload(posts: list[dict]) -> dict:
    """Wrap posts in the feed's ``{"data": {"posts": [...]}}`` envelope."""
    return {"data": {"posts": posts}}


def make_response(payload: dict, *, raise_error: Exception | None = None) -> MagicMock:
    """Build a mock httpx.Response: ``.json()`` and ``.raise_for_status()`` are sync."""
    response = MagicMock()
    response.json = MagicMock(return_value=payload)
    response.raise_for_status = MagicMock(side_effect=raise_error)
    return response


def make_client(response: MagicMock | None = None, *, get_error: Exception | None = None) -> AsyncMock:
    """Build a mock httpx.AsyncClient whose ``.get`` is awaitable."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=response, side_effect=get_error)
    return client


# ---------------------------------------------------------------------------
# parse_ninegag
# ---------------------------------------------------------------------------

class TestParseNinegag:
    def test_photo_post_becomes_candidate(self):
        payload = make_payload([make_post(
            post_id="xyz",
            title="Cat meme",
            image_url="https://9cache.com/photo/xyz_700b.jpg",
        )])
        result = ninegag.parse_ninegag(payload)
        assert result == [MemeCandidate(
            key="9gag:xyz",
            image_url="https://9cache.com/photo/xyz_700b.jpg",
            caption="Cat meme",
        )]

    def test_skips_animated_and_video_posts(self):
        payload = make_payload([
            make_post(post_id="a", post_type="Animated"),
            make_post(post_id="b", post_type="Video"),
            make_post(post_id="c", post_type="Photo"),
        ])
        keys = [candidate.key for candidate in ninegag.parse_ninegag(payload)]
        assert keys == ["9gag:c"]

    def test_skips_nsfw_posts(self):
        payload = make_payload([
            make_post(post_id="naughty", nsfw=1),
            make_post(post_id="clean", nsfw=0),
        ])
        keys = [candidate.key for candidate in ninegag.parse_ninegag(payload)]
        assert keys == ["9gag:clean"]

    def test_skips_post_without_image_url(self):
        payload = make_payload([make_post(post_id="noimg", image_url=None)])
        assert ninegag.parse_ninegag(payload) == []

    def test_key_falls_back_to_image_url_when_id_missing(self):
        post = make_post(image_url="https://9cache.com/photo/fallback.jpg")
        del post["id"]
        payload = make_payload([post])
        result = ninegag.parse_ninegag(payload)
        assert result[0].key == "9gag:https://9cache.com/photo/fallback.jpg"

    def test_missing_title_yields_empty_caption(self):
        post = make_post()
        del post["title"]
        result = ninegag.parse_ninegag(make_payload([post]))
        assert result[0].caption == ""

    def test_empty_payload_returns_empty_list(self):
        assert ninegag.parse_ninegag({}) == []
        assert ninegag.parse_ninegag({"data": {}}) == []
        assert ninegag.parse_ninegag(make_payload([])) == []


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

class TestNinegagFetch:
    async def test_returns_parsed_candidates_on_success(self):
        payload = make_payload([make_post(post_id="ok")])
        client = make_client(make_response(payload))
        result = await ninegag.fetch(client)
        assert [candidate.key for candidate in result] == ["9gag:ok"]

    async def test_sends_browser_user_agent(self):
        client = make_client(make_response(make_payload([])))
        await ninegag.fetch(client)
        headers = client.get.call_args.kwargs["headers"]
        assert headers["User-Agent"].startswith("Mozilla/")

    async def test_http_error_returns_empty_list(self):
        error = httpx.HTTPStatusError("403", request=MagicMock(), response=MagicMock())
        client = make_client(make_response(make_payload([]), raise_error=error))
        assert await ninegag.fetch(client) == []

    async def test_network_error_returns_empty_list(self):
        client = make_client(get_error=httpx.ConnectError("boom"))
        assert await ninegag.fetch(client) == []
