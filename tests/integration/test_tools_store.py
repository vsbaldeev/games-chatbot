"""Integration tests for Steam and PlayStation Store tools.

All tools in this module call public APIs that require no API keys.
Counter-Strike 2 is used as the reference game because it has a stable
Steam appid (730) and consistently high player counts.
"""

import json

import pytest

from src.tools.store import (
    get_ps_store_price_tr,
    get_ps_store_sales,
    get_steam_app_details,
    get_steam_player_count,
    get_steam_reviews_summary,
)


@pytest.mark.integration
class TestGetSteamPlayerCount:
    """Tests for get_steam_player_count against the real Steam API."""

    async def test_returns_player_data_for_known_game(self):
        """Counter-Strike 2 must return game name, appid, and a player count.

        current_players may be zero during maintenance but must never be absent.
        """
        raw = await get_steam_player_count.ainvoke({"game_name": "Counter-Strike 2"})
        result = json.loads(raw)
        assert "error" not in result
        assert "game" in result
        assert "appid" in result
        assert "current_players" in result
        assert result["appid"] > 0
        assert result["current_players"] >= 0

    async def test_unknown_game_returns_error_json(self):
        """A name that has no Steam match must produce a JSON error, not raise."""
        raw = await get_steam_player_count.ainvoke({"game_name": "xyzzy_no_such_game_9999"})
        result = json.loads(raw)
        assert "error" in result


@pytest.mark.integration
class TestGetSteamAppDetails:
    """Tests for get_steam_app_details against the real Steam API."""

    async def test_returns_details_for_known_game(self):
        """Counter-Strike 2 details must include name, developers, genres, steam_url."""
        raw = await get_steam_app_details.ainvoke({"game_name": "Counter-Strike 2"})
        result = json.loads(raw)
        assert "error" not in result
        assert "name" in result
        assert "developers" in result
        assert "genres" in result
        assert "steam_url" in result
        assert "steampowered.com" in result["steam_url"]
        assert isinstance(result["developers"], list)
        assert len(result["developers"]) > 0
        assert isinstance(result["genres"], list)

    async def test_unknown_game_returns_error_json(self):
        """A name not on Steam must produce a JSON error dict, not raise."""
        raw = await get_steam_app_details.ainvoke({"game_name": "xyzzy_no_such_game_9999"})
        result = json.loads(raw)
        assert "error" in result


@pytest.mark.integration
class TestGetSteamReviewsSummary:
    """Tests for get_steam_reviews_summary against the real Steam API."""

    async def test_returns_review_data_for_known_game(self):
        """Counter-Strike 2 reviews must include score label and review counts."""
        raw = await get_steam_reviews_summary.ainvoke({"game_name": "Counter-Strike 2"})
        result = json.loads(raw)
        assert "error" not in result
        assert "review_score_desc" in result
        assert "total_reviews" in result
        assert "total_positive" in result
        assert "total_negative" in result
        assert result["total_reviews"] > 0
        assert isinstance(result["review_score_desc"], str)
        assert len(result["review_score_desc"]) > 0


@pytest.mark.integration
@pytest.mark.xfail(strict=False, reason="psdeals.net RSS feed returns 403; scraping may be blocked by bot protection")
class TestGetPsStoreSales:
    """Tests for get_ps_store_sales against the psdeals.net RSS feed.

    Marked xfail because psdeals.net applies bot protection that blocks the
    request with 403 Forbidden. Tests pass when the feed is accessible and
    fail gracefully when it is not.
    """

    async def test_returns_a_list(self):
        """get_ps_store_sales must return a JSON list, not an error dict.

        The list may be empty when no active sales exist, but the response
        must always be a list, never a dict with an error key.
        """
        raw = await get_ps_store_sales.ainvoke({"limit": 12})
        result = json.loads(raw)
        assert isinstance(result, list)

    async def test_non_empty_results_are_title_strings(self):
        """Each sale entry must be a non-empty title string, not a nested object."""
        raw = await get_ps_store_sales.ainvoke({"limit": 5})
        result = json.loads(raw)
        for item in result:
            assert isinstance(item, str)
            assert len(item.strip()) > 0


@pytest.mark.integration
class TestGetPsStorePriceTr:
    """Tests for get_ps_store_price_tr against psdeals.net."""

    async def test_always_returns_ps_store_url(self):
        """ps_store_search_url must always be present regardless of price availability.

        The tool is designed never to raise — it always returns the search URL
        as a fallback even when the psdeals.net scrape fails.
        """
        raw = await get_ps_store_price_tr.ainvoke({"game_name": "Elden Ring"})
        result = json.loads(raw)
        assert "ps_store_search_url" in result
        assert "store.playstation.com" in result["ps_store_search_url"]

    async def test_store_url_encodes_the_game_name(self):
        """The generated URL must encode the searched game name."""
        raw = await get_ps_store_price_tr.ainvoke({"game_name": "Elden Ring"})
        result = json.loads(raw)
        store_url = result["ps_store_search_url"].lower()
        assert "elden" in store_url
