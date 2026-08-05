"""Integration tests for IGDB game database tools.

Calls the real IGDB API via Twitch OAuth. All tests are automatically skipped
when TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET are test stubs (i.e. the .env
file does not contain real credentials under those key names).

To enable: add TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET to .env.
Note: the current .env uses TWITCH_CLIEND_ID / TWITCH_SECRET (different names).
"""

import json
import os

import pytest

from src.tools.igdb import get_game_details, get_ps5_recommendations, search_games

SKIP_IGDB = (
    os.environ.get("TWITCH_CLIENT_ID", "test-twitch-id") == "test-twitch-id"
    or os.environ.get("TWITCH_CLIENT_SECRET", "test-twitch-secret") == "test-twitch-secret"
)
SKIP_REASON = "Real Twitch credentials not configured (TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET missing from .env)"


@pytest.mark.integration
@pytest.mark.skipif(SKIP_IGDB, reason=SKIP_REASON)
class TestSearchGames:
    """Tests for search_games against the real IGDB API."""

    async def test_returns_non_empty_list_for_known_title(self):
        """search_games must find at least one result for a well-known game.

        Elden Ring is a universally indexed, award-winning 2022 title and will
        always be present in IGDB.
        """
        raw = await search_games.ainvoke({"query": "Elden Ring"})
        games = json.loads(raw)
        assert isinstance(games, list)
        assert len(games) > 0

    async def test_each_result_has_required_fields(self):
        """Every result dict must contain the fields the pipeline depends on.

        The worker uses id to call get_game_details; name/rating/platforms/year
        are returned to the LLM for context.
        """
        raw = await search_games.ainvoke({"query": "Elden Ring"})
        games = json.loads(raw)
        for game in games:
            assert "id" in game
            assert "name" in game
            assert isinstance(game["id"], int)
            assert isinstance(game["name"], str)

    async def test_known_title_appears_in_results(self):
        """The queried game must appear in the results by name."""
        raw = await search_games.ainvoke({"query": "Elden Ring"})
        games = json.loads(raw)
        names = [game["name"] for game in games]
        assert any("Elden Ring" in name for name in names)


@pytest.mark.integration
@pytest.mark.skipif(SKIP_IGDB, reason=SKIP_REASON)
class TestGetGameDetails:
    """Tests for get_game_details against the real IGDB API."""

    async def test_returns_full_details_for_real_game(self):
        """get_game_details must return all expected fields for a real IGDB ID.

        Gets the ID dynamically via search_games so the test is not coupled
        to a hardcoded IGDB numeric ID that could change.
        """
        search_raw = await search_games.ainvoke({"query": "Elden Ring"})
        game_id = str(json.loads(search_raw)[0]["id"])

        raw = await get_game_details.ainvoke({"game_id": game_id})
        details = json.loads(raw)
        assert "error" not in details
        assert "name" in details
        assert "platforms" in details
        assert "genres" in details
        assert "game_modes" in details
        assert isinstance(details["platforms"], list)
        assert len(details["platforms"]) > 0

    async def test_nonexistent_id_returns_error_json(self):
        """A non-existent game ID must return a JSON error dict, not raise."""
        raw = await get_game_details.ainvoke({"game_id": "999999999"})
        result = json.loads(raw)
        assert "error" in result


@pytest.mark.integration
@pytest.mark.skipif(SKIP_IGDB, reason=SKIP_REASON)
class TestGetPs5Recommendations:
    """Tests for get_ps5_recommendations against the real IGDB API."""

    async def test_multiplayer_mode_returns_game_list(self):
        """Multiplayer mode must return a list of PS5 games with name and rating."""
        raw = await get_ps5_recommendations.ainvoke({"mode": "multiplayer"})
        games = json.loads(raw)
        assert isinstance(games, list)
        for game in games:
            assert "name" in game
            assert "rating" in game
            assert isinstance(game["name"], str)

    async def test_singleplayer_mode_returns_non_error_list(self):
        """Singleplayer mode must produce a valid JSON list, not an error dict."""
        raw = await get_ps5_recommendations.ainvoke({"mode": "singleplayer"})
        result = json.loads(raw)
        assert isinstance(result, list)
