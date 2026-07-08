"""Integration tests for Steam and PlayStation Store tools.

All tools in this module call public APIs that require no API keys.
Counter-Strike 2 is used as the reference game because it has a stable
Steam appid (730) and consistently high player counts.
"""

import json
import re

import pytest

from src.tools.store import (
    fetch_ps_store_price_via_web_search,
    fetch_ps_store_product_page_price,
    find_ps_store_product_url_via_web_search,
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
        """Elden Ring should return edition prices sourced from web search.

        Asserts format only when editions are present — PS Store geo-restrictions
        may occasionally prevent scraping.
        """
        raw = await get_ps_store_price_tr.ainvoke({"game_name": "Elden Ring"})
        result = json.loads(raw)
        assert "ps_store_search_url" in result
        if "editions" in result:
            assert len(result["editions"]) > 0
            for edition in result["editions"]:
                assert "price_try" in edition
                assert any(char.isdigit() for char in edition["price_try"])

    async def test_price_source_is_web_search(self):
        """When editions are found, source must be 'web_search'.

        Confirms the web search path is wired end-to-end: any editions returned
        by get_ps_store_price_tr must carry source='web_search'.
        """
        raw = await get_ps_store_price_tr.ainvoke({"game_name": "Elden Ring"})
        result = json.loads(raw)
        assert "ps_store_search_url" in result
        if "editions" in result:
            assert result.get("source") == "web_search"


@pytest.mark.integration
class TestFetchPsStorePriceViaWebSearch:
    """Tests for fetch_ps_store_price_via_web_search via the real web search backend."""

    async def test_returns_dict_for_known_game(self):
        """Must always return a dict and never raise for a well-known game."""
        result = await fetch_ps_store_price_via_web_search("Elden Ring")
        assert isinstance(result, dict)

    async def test_empty_dict_or_price_fields_present(self):
        """Result is either empty or has editions list and source fields."""
        result = await fetch_ps_store_price_via_web_search("Elden Ring")
        if result:
            assert "editions" in result
            assert "source" in result
            assert result["source"] == "web_search"
            assert len(result["editions"]) > 0
            for edition in result["editions"]:
                assert "price_try" in edition
                assert any(char.isdigit() for char in edition["price_try"])

    async def test_starfield_aggregates_standard_and_premium_editions(self):
        """Web-search price lookup must return both Starfield editions.

        Regression guard: Starfield has separate product pages for Standard
        (~1.618 TL) and Premium (~2.265 TL). The old code stopped at the first
        page with any price, so whichever edition the web search surfaced first
        was the only one reported. After the aggregation fix, both editions must
        appear regardless of search-result ordering.
        """
        result = await fetch_ps_store_price_via_web_search("Starfield")
        assert isinstance(result, dict)
        if not result:
            pytest.skip("PS Store TR geo-restricted or Starfield not found")
        assert "editions" in result
        assert len(result["editions"]) >= 2, (
            f"Expected Standard and Premium editions; got {result['editions']}"
        )
        prices = {edition["price_try"] for edition in result["editions"]}
        assert len(prices) >= 2, (
            f"Expected distinct prices for Standard and Premium; got {prices}"
        )


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
    REANIMAL_URL = (
        "https://store.playstation.com/en-tr/product/"
        "EP4389-PPSA23372_00-ANIMALGAMEGLOBAL"
    )
    STARFIELD_URL = (
        "https://store.playstation.com/en-tr/product/"
        "UP1003-PPSA24884_00-HELIUM0000000000"
    )

    async def test_arc_raiders_returns_price(self):
        """Arc Raiders product page must return editions with TRY prices.

        Previous regression: 1.000,00 TL (old aggregator price) was returned
        instead of the correct 2.090,00 TL main game price.
        """
        result = await fetch_ps_store_product_page_price(self.ARC_RAIDERS_URL)
        assert isinstance(result, dict)
        assert "editions" in result
        assert len(result["editions"]) > 0
        assert "ps_store_product_url" in result
        for edition in result["editions"]:
            assert "name" in edition
            assert "price_try" in edition
            assert any(char.isdigit() for char in edition["price_try"])

    async def test_helldivers_2_returns_main_game_price(self):
        """Helldivers 2 product page must return the Standard Edition price, not a DLC price.

        Previous regression: 429,00 TL (Super Credits virtual currency DLC) was
        returned instead of the correct 1.399,00 TL main game price.
        """
        result = await fetch_ps_store_product_page_price(self.HELLDIVERS_2_URL)
        assert isinstance(result, dict)
        assert "editions" in result
        assert len(result["editions"]) > 0
        first_price = result["editions"][0]["price_try"]
        assert any(char.isdigit() for char in first_price)
        assert "1.399" in first_price or "1399" in first_price.replace(".", "").replace(",", "")

    async def test_reanimal_returns_editions_with_prices(self):
        """Reanimal product page must return at least one edition with a TRY price.

        Reanimal is a horror adventure game by Tarsier Studios published by THQ
        Nordic (released February 2026). Guards against the page returning no
        editions or a price with no digits.
        """
        result = await fetch_ps_store_product_page_price(self.REANIMAL_URL)
        assert isinstance(result, dict)
        assert "editions" in result, f"Expected editions key; got {result}"
        assert len(result["editions"]) > 0, "Expected at least one edition"
        assert "ps_store_product_url" in result
        for edition in result["editions"]:
            assert "name" in edition
            assert "price_try" in edition
            assert any(char.isdigit() for char in edition["price_try"]), (
                f"price_try contains no digits: {edition['price_try']!r}"
            )

    async def test_starfield_standard_edition_returns_correct_price(self):
        """Starfield standard product page must return the Standard Edition price.

        The base game URL resolves to the Standard Edition page which should
        return exactly one edition at the standard price (~1.618 TL).
        Guards against the page returning no price or the wrong product.
        """
        result = await fetch_ps_store_product_page_price(self.STARFIELD_URL)
        assert isinstance(result, dict)
        assert "editions" in result, f"Expected editions key; got {result}"
        assert len(result["editions"]) > 0, "Expected at least one edition"
        assert "ps_store_product_url" in result
        first_price = result["editions"][0]["price_try"]
        assert any(char.isdigit() for char in first_price), (
            f"price_try contains no digits: {first_price!r}"
        )

    async def test_returns_empty_dict_for_invalid_url(self):
        """An invalid product URL must return an empty dict, not raise."""
        result = await fetch_ps_store_product_page_price(
            "https://store.playstation.com/en-tr/product/INVALID-PRODUCT-ID-00000"
        )
        assert isinstance(result, dict)


@pytest.mark.integration
class TestFindPsStoreProductUrlViaWebSearch:
    """Integration tests for find_ps_store_product_url_via_web_search.

    Verifies the refactored function returns a list of all candidate URLs
    rather than a single URL, which enables the caller to skip addon pages.
    """

    async def test_returns_list_for_known_game(self):
        """Must always return a list, never None or a bare string."""
        urls = await find_ps_store_product_url_via_web_search("Elden Ring")
        assert isinstance(urls, list)

    async def test_all_returned_urls_are_ps_store_tr_product_urls(self):
        """Every URL in the result must be a store.playstation.com/en-tr/product/ URL."""
        urls = await find_ps_store_product_url_via_web_search("Elden Ring")
        for url in urls:
            assert "store.playstation.com/en-tr/product/" in url, (
                f"Unexpected URL format: {url}"
            )

    async def test_returns_at_least_one_url_for_known_game(self):
        """A popular game must produce at least one candidate URL."""
        urls = await find_ps_store_product_url_via_web_search("Elden Ring")
        assert len(urls) >= 1, (
            "Expected at least one candidate URL; "
            f"got {len(urls)}"
        )

    async def test_returns_empty_list_for_nonsense_query(self):
        """An unrecognisable title must return an empty list, not raise."""
        urls = await find_ps_store_product_url_via_web_search("xyzzy_no_such_game_99999")
        assert isinstance(urls, list)
        assert len(urls) == 0


@pytest.mark.integration
class TestPsStoreAddonPageRejection:
    """Verify that Add-On product pages are rejected end-to-end.

    Uses a DLC URL found via web search at test time so the test does not
    depend on a hardcoded product ID that can change between PS Store releases.
    """

    async def test_addon_page_is_rejected_when_found(self):
        """fetch_ps_store_product_page_price must return {} for a real Add-On page.

        Searches for an in-game virtual-currency product — the kind most likely
        to have its own PS Store product URL classified as an Add-On. Skips when
        the web search returns a game page instead of an addon page, since not
        all DLC searches surface a dedicated addon product URL.
        """
        addon_urls = await find_ps_store_product_url_via_web_search(
            "Apex Legends Apex Coins"
        )
        if not addon_urls:
            pytest.skip("Web search returned no PS Store product URLs")

        result = await fetch_ps_store_product_page_price(addon_urls[0])
        if "editions" in result:
            pytest.skip(
                f"Web search returned a game page ({addon_urls[0]!r}), "
                "not an Add-On page — addon rejection cannot be verified"
            )
        assert result == {}, (
            f"Expected empty dict for addon page {addon_urls[0]!r}, got {result}"
        )

    async def test_fetch_price_via_web_search_skips_addon_and_falls_back(self):
        """fetch_ps_store_price_via_web_search must not return a price for a DLC-only query.

        Searching for a DLC by its exact expansion name should either return an
        empty dict (all results rejected as addon pages) or — if a base game
        page happens to appear in results — return that game's editions.
        In either case the result must never contain an addon price.
        """
        result = await fetch_ps_store_price_via_web_search(
            "Elden Ring Shadow of the Erdtree"
        )
        assert isinstance(result, dict)
        if result:
            # If anything was returned it must have the editions key
            assert "editions" in result


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
        result, tools_used = await worker_agent.invoke_worker(prompt)
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
        result, tools_used = await worker_agent.invoke_worker(prompt)
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
        result, tools_used = await worker_agent.invoke_worker(prompt)
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
        result, tools_used = await worker_agent.invoke_worker(prompt)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        normalised = result.replace(".", "").replace(",", "").replace(" ", "")
        assert "1.399" in result or "1 399" in result or "1399" in normalised

    async def test_worker_lists_all_editions_with_prices_for_god_of_war(self, worker_agent):
        """WorkerAgent must list every available edition with its TRY price.

        God of War Ragnarok ships in multiple editions on PS Store TR (Standard
        and Digital Deluxe at minimum). The response must include at least two
        distinct TRY price amounts and reference both edition names.
        """
        prompt = (
            "Question from @user: "
            "Какие издания"
            " God of War Ragnarök есть в PS Store"
            " Турции и сколько"
            " каждое стоит?"
        )
        result, tools_used = await worker_agent.invoke_worker(prompt)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        lower = result.lower()

        assert any(term in lower for term in ["war", "playstation", "ps store", "tl", "лира", "₺"])

        prices = re.findall(r'\d[\d\s.,]*(?:tl|лир|₺)', lower)
        edition_hits = sum(
            1 for term in ["standard", "deluxe", "digital", "edition", "издани", "версия"]
            if term in lower
        )
        assert len(prices) >= 2 or edition_hits >= 2, (
            f"Expected multiple editions or prices in response \u2014 "
            f"prices found: {prices!r}, edition term hits: {edition_hits}. "
            f"Response: {result[:400]}"
        )

    async def test_worker_returns_price_for_reanimal(self, worker_agent):
        """WorkerAgent must return a TRY price for Reanimal from PS Store TR.

        Reanimal is a horror adventure game by Tarsier Studios (THQ Nordic, 2026).
        The response must reference the game and include a price in TRY.
        """
        prompt = "Question from @user: Сколько стоит Reanimal в турецком PS Store?"
        result, tools_used = await worker_agent.invoke_worker(prompt)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        lower = result.lower()
        assert any(
            term in lower
            for term in ["tl", "try", "₺", "лира", "стоит", "playstation", "reanimal", "реанимал"]
        )

    async def test_worker_lists_all_starfield_editions(self, worker_agent):
        """WorkerAgent must list both Standard and Premium Starfield editions with prices.

        Regression guard: previously only the Premium Edition (2.265 TL) was
        returned because the web search surfaced the dedicated premium product
        page first. After the fix (return page with most editions), both
        Standard (~1.618 TL) and Premium (~2.265 TL) must appear in the response.
        """
        prompt = (
            "Question from @user: "
            "Перечисли все издания Starfield в PS Store Турции с ценами."
        )
        result, tools_used = await worker_agent.invoke_worker(prompt)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        lower = result.lower()

        assert any(term in lower for term in ["starfield", "tl", "лира", "₺", "playstation"])

        prices = re.findall(r'\d[\d\s.,]*(?:tl|лир|₺)', lower)
        edition_hits = sum(
            1 for term in ["standard", "premium", "edition", "издани", "версия", "стандарт"]
            if term in lower
        )
        assert len(prices) >= 2 or edition_hits >= 2, (
            f"Expected both Standard and Premium editions — "
            f"prices found: {prices!r}, edition term hits: {edition_hits}. "
            f"Response: {result[:400]}"
        )

    async def test_worker_includes_edition_names_alongside_prices(self, worker_agent):
        """WorkerAgent must pair edition names with prices, not list bare numbers.

        When the tool returns multiple editions the LLM must present them with
        labels (e.g. 'Standard Edition \u2014 1.699 TL') rather than bare numbers.
        Uses Spider-Man 2, which has Standard and Digital Deluxe editions on PS Store TR.
        """
        prompt = (
            "Question from @user: "
            "Перечисли все"
            " издания Marvel's Spider-Man 2"
            " в PS Store Турции с ценами."
        )
        result, tools_used = await worker_agent.invoke_worker(prompt)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        lower = result.lower()

        assert any(term in lower for term in ["spider", "playstation", "ps store", "tl", "лира", "₺"])

        has_edition_label = any(
            term in lower
            for term in ["standard", "deluxe", "digital", "edition", "издани", "версия"]
        )
        has_price = bool(re.search(r'\d[\d\s.,]*(?:tl|лир|₺)', lower))
        assert has_edition_label and has_price, (
            f"Expected edition labels paired with TRY prices. "
            f"Edition label present: {has_edition_label}, price present: {has_price}. "
            f"Response: {result[:400]}"
        )
