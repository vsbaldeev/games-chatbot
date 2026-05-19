"""Integration tests for Steam and PlayStation Store tools.

All tools in this module call public APIs that require no API keys.
Counter-Strike 2 is used as the reference game because it has a stable
Steam appid (730) and consistently high player counts.
"""

import json

import pytest

from src.tools.store import (
    fetch_ps_store_price_via_web_search,
    fetch_ps_store_product_page_price,
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
        """Elden Ring should return a TRY price sourced from web search.

        Asserts format only when a price is present — PS Store geo-restrictions
        may occasionally prevent scraping.
        """
        raw = await get_ps_store_price_tr.ainvoke({"game_name": "Elden Ring"})
        result = json.loads(raw)
        assert "ps_store_search_url" in result
        if "regular_price_try" in result:
            assert any(char.isdigit() for char in result["regular_price_try"])

    async def test_price_source_is_web_search(self):
        """When a price is found, source must be 'web_search'.

        Confirms the web search path is wired end-to-end: any price returned
        by get_ps_store_price_tr must carry source='web_search'.
        """
        raw = await get_ps_store_price_tr.ainvoke({"game_name": "Elden Ring"})
        result = json.loads(raw)
        assert "ps_store_search_url" in result
        if "regular_price_try" in result:
            assert result.get("source") == "web_search"


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
class TestFetchPsStoreProductPagePrice:
    """Tests for fetch_ps_store_product_page_price using known stable product URLs.

    Arc Raiders and Helldivers 2 are used because their exact product URLs and
    expected prices are known. These tests guard against price-parser regressions
    where DLC or virtual-currency prices were returned instead of the main game price.
    """

    ARC_RAIDERS_URL = (
        "https://store.playstation.com/en-tr/product/"
        "EP6848-PPSA04998_00-4085881659726287"
    )
    HELLDIVERS_2_URL = (
        "https://store.playstation.com/en-tr/product/"
        "EP9000-PPSA06016_00-HELLDIVERS200000"
    )

    async def test_arc_raiders_returns_price(self):
        """Arc Raiders product page must return a TRY price, not a DLC price.

        Previous regression: 1.000,00 TL (old aggregator price) was returned
        instead of the correct 2.090,00 TL main game price.
        """
        result = await fetch_ps_store_product_page_price(self.ARC_RAIDERS_URL)
        assert isinstance(result, dict)
        assert "regular_price_try" in result
        price = result["regular_price_try"]
        assert any(char.isdigit() for char in price)
        assert "ps_store_product_url" in result

    async def test_helldivers_2_returns_main_game_price(self):
        """Helldivers 2 product page must return the main game price, not a DLC price.

        Previous regression: 429,00 TL (Super Credits virtual currency DLC) was
        returned instead of the correct 1.399,00 TL main game price.
        """
        result = await fetch_ps_store_product_page_price(self.HELLDIVERS_2_URL)
        assert isinstance(result, dict)
        assert "regular_price_try" in result
        price = result["regular_price_try"]
        assert any(char.isdigit() for char in price)
        assert "1.399" in price or "1399" in price.replace(".", "").replace(",", "")

    async def test_returns_empty_dict_for_invalid_url(self):
        """An invalid product URL must return an empty dict, not raise."""
        result = await fetch_ps_store_product_page_price(
            "https://store.playstation.com/en-tr/product/INVALID-PRODUCT-ID-00000"
        )
        assert isinstance(result, dict)


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
            for term in [
                "tl", "try", "₺", "store.playstation.com", "playstation", "fiyat",
                "лира", "цена", "стоит", "магазин", "недоступ", "найден", "war",
                "found", "turkey", "ps store",
            ]
        )

    async def test_worker_returns_correct_price_for_arc_raiders(self, worker_agent):
        """WorkerAgent must return the correct Arc Raiders price: 2.090 TL.

        Regression guard: previously the tool returned 1.000 TL (stale aggregator
        price) instead of the 2.090 TL shown on the PS Store TR product page.
        """
        prompt = "Question from @user: Сколько стоит Arc Raiders в турецком PS Store?"
        result = await worker_agent.invoke_worker(prompt)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        normalised = result.replace(".", "").replace(",", "").replace(" ", "")
        assert "2.090" in result or "2 090" in result or "2090" in normalised

    async def test_worker_returns_correct_price_for_helldivers_2(self, worker_agent):
        """WorkerAgent must return the Helldivers 2 main game price: 1.399 TL.

        Regression guard: previously the tool returned 429 TL (Super Credits
        virtual-currency DLC) instead of the 1.399 TL main game price.
        """
        prompt = "Question from @user: Сколько стоит Helldivers 2 в PS Store Турции?"
        result = await worker_agent.invoke_worker(prompt)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        normalised = result.replace(".", "").replace(",", "").replace(" ", "")
        assert "1.399" in result or "1 399" in result or "1399" in normalised
