"""Integration tests for media tools: AniList and TMDB.

search_anime calls the public AniList API — no key required.
search_movie_or_tv requires TMDB_API_KEY and is skipped when absent.
"""

import json
import os

import pytest

from src import config
from src.tools.media import search_anime, search_movie_or_tv

SKIP_TMDB = not os.environ.get("TMDB_API_KEY")
SKIP_TMDB_REASON = "TMDB_API_KEY not configured in .env"


@pytest.mark.integration
class TestSearchAnime:
    """Tests for search_anime against the real AniList GraphQL API."""

    async def test_returns_details_for_known_title(self):
        """Demon Slayer must return title, episodes, status, score, and studios.

        Demon Slayer (Kimetsu no Yaiba) is a completed, widely indexed series
        with stable metadata — a reliable reference title.
        """
        raw = await search_anime.ainvoke({"query": "Demon Slayer"})
        result = json.loads(raw)
        assert "error" not in result
        assert "title_romaji" in result
        assert "episodes" in result
        assert "status" in result
        assert "average_score" in result
        assert "studios" in result
        assert isinstance(result["studios"], list)
        assert len(result["studios"]) > 0

    async def test_unknown_title_returns_error_json(self):
        """A search with no AniList match must return a JSON error, not raise."""
        raw = await search_anime.ainvoke({"query": "xyzzy_no_such_anime_abc_99999"})
        result = json.loads(raw)
        assert "error" in result


@pytest.mark.integration
class TestSearchMovieOrTv:
    """Tests for search_movie_or_tv against TMDB and for the no-key fallback."""

    async def test_without_api_key_returns_config_error(self, monkeypatch):
        """When TMDB_API_KEY is empty the tool must return a config error, not raise."""
        monkeypatch.setattr(config, "TMDB_API_KEY", "")
        raw = await search_movie_or_tv.ainvoke({"query": "Inception", "media_type": "movie"})
        result = json.loads(raw)
        assert "error" in result
        assert "TMDB_API_KEY" in result["error"]

    @pytest.mark.skipif(SKIP_TMDB, reason=SKIP_TMDB_REASON)
    async def test_returns_movie_details_for_known_title(self):
        """Inception must return title, year, genres, vote_average, and tmdb_url."""
        raw = await search_movie_or_tv.ainvoke({"query": "Inception", "media_type": "movie"})
        result = json.loads(raw)
        assert "error" not in result
        assert "title" in result
        assert "year" in result
        assert "genres" in result
        assert "vote_average" in result
        assert "tmdb_url" in result
        assert "themoviedb.org" in result["tmdb_url"]
        assert len(result["year"]) == 4

    @pytest.mark.skipif(SKIP_TMDB, reason=SKIP_TMDB_REASON)
    async def test_tv_search_returns_series_details(self):
        """Searching for a TV series must return results with the same shape as movies."""
        raw = await search_movie_or_tv.ainvoke({"query": "Breaking Bad", "media_type": "tv"})
        result = json.loads(raw)
        assert "error" not in result
        assert "title" in result
        assert "vote_average" in result
        assert result["vote_average"] > 0
