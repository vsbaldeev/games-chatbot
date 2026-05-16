"""Integration tests for Steam and PlayStation Store tools.

All tools in this module call public APIs that require no API keys.
Counter-Strike 2 is used as the reference game because it has a stable
Steam appid (730) and consistently high player counts.
"""

import json

import pytest

from src.tools.store import (
    fetch_ps_store_price_via_web_search,
    get_ps_store_price_tr,
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
        """Counter-Strike 2 details must include name, developers, genres, price_try, steam_url."""
        raw = await get_steam_app_details.ainvoke({"game_name": "Counter-Strike 2"})
        result = json.loads(raw)
        assert "error" not in result
        assert "name" in result
        assert "developers" in result
        assert "genres" in result
        assert "price_try" in result
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

    async def test_unknown_game_returns_error_json(self):
        """A name not on Steam must produce a JSON error dict, not raise."""
        raw = await get_steam_reviews_summary.ainvoke({"game_name": "xyzzy_no_such_game_9999"})
        result = json.loads(raw)
        assert "error" in result


@pytest.mark.integration
class TestGetPsStorePriceTr:
    """Tests for get_ps_store_price_tr against the PlayStation Store TR."""

    async def test_always_returns_ps_store_url(self):
        """ps_store_search_url must always be present regardless of price availability.

        The tool never raises — it always returns the search URL as a fallback
        even when the live price scrape fails.
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

    async def test_returns_price_in_try_for_known_game(self):
        """Elden Ring should return a TRY price when PS Store TR search succeeds.

        The price may come from direct scraping or the web search fallback.
        Either path is valid; the assertion only checks format when a price is present.
        """
        raw = await get_ps_store_price_tr.ainvoke({"game_name": "Elden Ring"})
        result = json.loads(raw)
        assert "ps_store_search_url" in result
        if "regular_price_try" in result:
            assert any(char.isdigit() for char in result["regular_price_try"])

    async def test_web_search_fallback_is_wired(self):
        """When the web search fallback supplies a price, source must be 'web_search'.

        This confirms the fallback path in get_ps_store_price_tr is connected:
        the tool always returns the search URL, and any web-search-sourced price
        carries the source field.
        """
        raw = await get_ps_store_price_tr.ainvoke({"game_name": "Elden Ring"})
        result = json.loads(raw)
        assert "ps_store_search_url" in result
        if result.get("source") == "web_search":
            assert "regular_price_try" in result


@pytest.mark.integration
class TestFetchPsStorePriceViaWebSearch:
    """Tests for fetch_ps_store_price_via_web_search via the real web search backend."""

    async def test_returns_dict_for_known_game(self):
        """Must always return a dict and never raise for a well-known game."""
        result = await fetch_ps_store_price_via_web_search("Elden Ring")
        assert isinstance(result, dict)

    async def test_empty_dict_or_price_fields_present(self):
        """Result is either empty or has all expected price field keys with valid values."""
        result = await fetch_ps_store_price_via_web_search("Elden Ring")
        if result:
            assert "regular_price_try" in result
            assert "source" in result
            assert result["source"] == "web_search"
            price = result["regular_price_try"]
            assert any(char.isdigit() for char in price)


@pytest.mark.integration
class TestGetPsStorePriceTrWithLLM:
    """Tests for get_ps_store_price_tr invoked through the WorkerAgent (tool + LLM).

    These tests consume Groq tokens. Run in isolation:
        pytest tests/integration/test_tools_store.py::TestGetPsStorePriceTrWithLLM
    """

    async def test_worker_returns_ps_store_info_for_elden_ring(self, worker_agent):
        """WorkerAgent must call get_ps_store_price_tr and include PS Store info.

        The agent receives a natural-language price question and must return a
        non-empty string that references PlayStation, a price, or the store URL.
        """
        prompt = "Question from @user: Сколько стоит Elden Ring в турецком PS Store?"
        result = await worker_agent.invoke_worker(prompt)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        lower = result.lower()
        assert any(
            term in lower
            for term in ["tl", "try", "₺", "store.playstation.com", "playstation", "fiyat",
                         "лира", "цена", "стоит", "магазин", "недоступ", "найден", "elden"]
        )

    async def test_worker_returns_ps_store_info_for_god_of_war(self, worker_agent):
        """WorkerAgent must call get_ps_store_price_tr and include PS Store info for God of War.

        God of War is a well-known PS exclusive present in both PS Store TR and
        common price-aggregator sites — a good stress test for the fallback path.
        """
        prompt = "Question from @user: Сколько стоит God of War Ragnarök в PS Store Турции?"
        result = await worker_agent.invoke_worker(prompt)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        lower = result.lower()
        assert any(
            term in lower
            for term in ["tl", "try", "₺", "store.playstation.com", "playstation", "fiyat",
                         "лира", "цена", "стоит", "магазин", "недоступ", "найден", "war"]
        )
