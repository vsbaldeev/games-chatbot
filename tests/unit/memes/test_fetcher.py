"""Unit tests for the meme fetcher orchestration.

Covers:
  - gather_candidates: fan-out over the SOURCES registry
  - get_meme: per-chat deduplication plus the vision-gate vetting loop
  - download_image: byte download used to upload memes Telegram can't fetch
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.memes import fetcher
from src.memes.sources.base import MemeCandidate

CHAT_ID = 1000
PASS_SCORE = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def candidate(key: str, *, url: str = "") -> MemeCandidate:
    """Build a MemeCandidate, defaulting the image URL from the key."""
    return MemeCandidate(key=key, image_url=url or f"https://img/{key}.jpg")


def source_returning(*candidates: MemeCandidate):
    """Build a fake async source fetcher that ignores its client argument."""
    async def fetch(client):
        return list(candidates)
    return fetch


@contextmanager
def vetting(*, candidates, seen=frozenset(), downloads=b"IMG", scores):
    """Patch every collaborator of the get_meme vetting loop.

    Args:
        candidates: What gather_candidates returns.
        seen: Keys already sent to this chat.
        downloads: A single bytes value, or a list used as a side_effect
            sequence where None is a failed download.
        scores: Judge verdicts as a side_effect sequence; None means the
            judge was unavailable.

    Yields:
        The patched mocks by name, so tests can assert on the calls.
    """
    download_kwargs = ({"side_effect": downloads} if isinstance(downloads, list)
                       else {"return_value": downloads})
    mocks = {
        "gather_candidates": AsyncMock(return_value=list(candidates)),
        "get_seen_urls": AsyncMock(return_value=set(seen)),
        "mark_seen": AsyncMock(),
        "download_image": AsyncMock(**download_kwargs),
        "score_meme": AsyncMock(side_effect=list(scores)),
    }
    with patch.multiple("src.memes.fetcher", **mocks):
        yield mocks


def seen_keys(mark_seen: AsyncMock) -> list[str]:
    """Keys passed to mark_seen, in call order."""
    return [call.args[1] for call in mark_seen.await_args_list]


# ---------------------------------------------------------------------------
# gather_candidates
# ---------------------------------------------------------------------------

class TestGatherCandidates:
    async def test_combines_all_sources(self):
        sources = (
            source_returning(candidate("9gag:a"), candidate("9gag:b")),
            source_returning(candidate("tg:chan/1")),
        )
        with patch("src.memes.fetcher.SOURCES", sources):
            result = await fetcher.gather_candidates()
        assert [item.key for item in result] == ["9gag:a", "9gag:b", "tg:chan/1"]

    async def test_no_sources_returns_empty(self):
        with patch("src.memes.fetcher.SOURCES", ()):
            assert await fetcher.gather_candidates() == []


# ---------------------------------------------------------------------------
# get_meme
# ---------------------------------------------------------------------------

class TestGetMeme:
    async def test_returns_vetted_image_bytes(self):
        with vetting(candidates=[candidate("9gag:x")], scores=[PASS_SCORE]) as mocks:
            result = await fetcher.get_meme(CHAT_ID)
        assert result == b"IMG"
        mocks["mark_seen"].assert_awaited_once_with(CHAT_ID, "9gag:x")

    async def test_filters_seen_keys_before_judging(self):
        candidates = [candidate("tg:chan/1"), candidate("tg:chan/2")]
        with vetting(candidates=candidates, seen={"tg:chan/1"}, scores=[PASS_SCORE]) as mocks:
            await fetcher.get_meme(CHAT_ID)
        assert seen_keys(mocks["mark_seen"]) == ["tg:chan/2"]
        mocks["download_image"].assert_awaited_once_with("https://img/tg:chan/2.jpg")

    async def test_rejected_candidate_is_burned_and_next_one_ships(self):
        candidates = [candidate("tg:junk"), candidate("tg:good")]
        with vetting(candidates=candidates, scores=[PASS_SCORE - 1, PASS_SCORE]) as mocks:
            result = await fetcher.get_meme(CHAT_ID)
        assert result == b"IMG"
        # Both marked: the reject permanently, so it never costs a second
        # download and vision call, and the winner because it was sent.
        assert set(seen_keys(mocks["mark_seen"])) == {"tg:junk", "tg:good"}

    async def test_every_candidate_rejected_returns_none(self):
        candidates = [candidate(f"tg:junk{index}") for index in range(3)]
        with vetting(candidates=candidates, scores=[0, 1, 2]) as mocks:
            assert await fetcher.get_meme(CHAT_ID) is None
        assert len(seen_keys(mocks["mark_seen"])) == 3

    async def test_stops_after_the_configured_attempt_budget(self):
        candidates = [candidate(f"tg:junk{index}") for index in range(10)]
        with vetting(candidates=candidates, scores=[0] * 10) as mocks:
            await fetcher.get_meme(CHAT_ID)
        assert mocks["score_meme"].await_count == 3

    async def test_judge_outage_sends_nothing_and_burns_nothing(self):
        """An outage is not a verdict: those candidates must stay available."""
        candidates = [candidate("tg:a"), candidate("tg:b")]
        with vetting(candidates=candidates, scores=[None, PASS_SCORE]) as mocks:
            assert await fetcher.get_meme(CHAT_ID) is None
        mocks["mark_seen"].assert_not_awaited()

    async def test_judge_outage_aborts_instead_of_retrying(self):
        candidates = [candidate("tg:a"), candidate("tg:b"), candidate("tg:c")]
        with vetting(candidates=candidates, scores=[None, None, None]) as mocks:
            await fetcher.get_meme(CHAT_ID)
        mocks["score_meme"].assert_awaited_once()

    async def test_download_failure_tries_next_without_burning_it(self):
        candidates = [candidate("tg:a"), candidate("tg:b")]
        with vetting(candidates=candidates, downloads=[None, b"IMG"],
                     scores=[PASS_SCORE]) as mocks:
            result = await fetcher.get_meme(CHAT_ID)
        assert result == b"IMG"
        assert len(seen_keys(mocks["mark_seen"])) == 1

    @pytest.mark.parametrize(
        "candidates, seen",
        [([], set()), ([candidate("9gag:a")], {"9gag:a"})],
        ids=["nothing-fetched", "everything-already-sent"],
    )
    async def test_empty_pool_returns_none_without_judging(self, candidates, seen):
        with vetting(candidates=candidates, seen=seen, scores=[]) as mocks:
            assert await fetcher.get_meme(CHAT_ID) is None
        mocks["score_meme"].assert_not_awaited()
        mocks["download_image"].assert_not_awaited()

    async def test_gather_failure_returns_none(self):
        with patch("src.memes.fetcher.gather_candidates", AsyncMock(side_effect=RuntimeError("boom"))):
            assert await fetcher.get_meme(CHAT_ID) is None


# ---------------------------------------------------------------------------
# download_image
# ---------------------------------------------------------------------------

def image_response(content: bytes, *, raise_error: Exception | None = None) -> MagicMock:
    """Build a mock httpx.Response with raw ``.content`` bytes."""
    response = MagicMock()
    response.content = content
    response.raise_for_status = MagicMock(side_effect=raise_error)
    return response


def client_context(client: AsyncMock) -> MagicMock:
    """Wrap a mock client so ``async with httpx.AsyncClient(...)`` yields it."""
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    return context


class TestDownloadImage:
    async def test_returns_bytes_on_success(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=image_response(b"JPEGDATA"))
        with patch("src.memes.fetcher.httpx.AsyncClient", return_value=client_context(client)):
            result = await fetcher.download_image("https://cdn4.telesco.pe/file/abc")
        assert result == b"JPEGDATA"

    async def test_sends_browser_user_agent(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=image_response(b"x"))
        with patch("src.memes.fetcher.httpx.AsyncClient", return_value=client_context(client)):
            await fetcher.download_image("https://cdn4.telesco.pe/file/abc")
        headers = client.get.call_args.kwargs["headers"]
        assert headers["User-Agent"].startswith("Mozilla/")

    async def test_http_error_returns_none(self):
        error = httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())
        client = AsyncMock()
        client.get = AsyncMock(return_value=image_response(b"", raise_error=error))
        with patch("src.memes.fetcher.httpx.AsyncClient", return_value=client_context(client)):
            assert await fetcher.download_image("https://cdn4.telesco.pe/file/abc") is None

    async def test_network_error_returns_none(self):
        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
        with patch("src.memes.fetcher.httpx.AsyncClient", return_value=client_context(client)):
            assert await fetcher.download_image("https://cdn4.telesco.pe/file/abc") is None
