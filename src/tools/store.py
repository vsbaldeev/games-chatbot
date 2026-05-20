"""PlayStation Store and Steam tools."""

import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx
from langchain_core.tools import tool


async def __find_steam_appid(game_name: str, client: httpx.AsyncClient) -> tuple[int, str] | None:
    """Search Steam for a game by name and return its appid and canonical name.

    Args:
        game_name: Human-readable game title to search for.
        client: Shared async HTTP client.

    Returns:
        A (appid, name) tuple for the top result, or None if not found.
    """
    response = await client.get(
        "https://store.steampowered.com/api/storesearch/",
        params={"term": game_name, "cc": "tr", "l": "en"},
    )
    response.raise_for_status()
    data = response.json()
    items = data.get("items", [])
    if not items:
        return None
    return items[0]["id"], items[0]["name"]


@tool
async def get_ps_store_sales(limit: int = 12) -> str:
    """
    Fetch current PS Store sale titles from the psdeals.net RSS feed.
    Returns a JSON list of discounted game title strings (up to `limit` entries).
    """
    try:
        safe_limit = max(1, min(limit, 50))
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(
                "https://psdeals.net/rss-feed",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
        root = ET.fromstring(response.text)
        titles = [
            item.text.strip()
            for item in root.findall(".//item/title")
            if item.text
        ][:safe_limit]
        return json.dumps(titles, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


def is_ps_store_addon_page(html: str) -> bool:
    """Return True if the PS Store page is for an Add-On rather than a base game.

    Checks the Next.js SSR data and rendered HTML for add-on category markers.
    Used to skip DLC/addon URLs returned by web search before scraping a price.

    Args:
        html: Full HTML of a store.playstation.com product page.

    Returns:
        True if the page is detected as an Add-On or DLC, False otherwise.
    """
    addon_signals = [
        r'"topCategory"\s*:\s*"(?:Add-On|ADD_ON|GAME_ADD_ON)"',
        r'"contentType"\s*:\s*"GAME_ADD_ON"',
        r'data-qa="[^"]*topCategory[^"]*"[^>]*>\s*Add-On',
    ]
    return any(re.search(pattern, html, re.IGNORECASE) for pattern in addon_signals)


def extract_offer_name(html: str, offer_index: str) -> str | None:
    """Extract the display name for one PS Store offer by its index.

    Tries known data-qa naming patterns used by the PS Store Next.js layer.

    Args:
        html: Full HTML of a store.playstation.com product page.
        offer_index: String index of the offer (e.g. ``"0"``, ``"1"``).

    Returns:
        The offer name, or None if no recognisable name element is found.
    """
    name_patterns = [
        rf'data-qa="mfeCtaMain#offer{offer_index}#label"[^>]*>(.*?)</(?:span|div|p)>',
        rf'data-qa="mfeCtaMain#offer{offer_index}#name"[^>]*>(.*?)</(?:span|div|p)>',
    ]
    for pattern in name_patterns:
        match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if match:
            name = re.sub(r'<[^>]+>', '', match.group(1)).strip()
            if name:
                return name
    return None


def scrape_ps_store_editions(html: str) -> list[dict]:
    """Extract all editions with names and prices from PS Store product page HTML.

    Finds every ``mfeCtaMain#offerN#finalPrice`` span and pairs each price with
    an edition name from ``extract_offer_name``.

    Args:
        html: Full HTML of a store.playstation.com/en-tr/product/... page.

    Returns:
        List of ``{"name": ..., "price_try": ...}`` dicts, one per edition.
        Empty list if no offer prices are found.
    """
    price_matches = re.findall(
        r'data-qa="mfeCtaMain#offer(\d+)#finalPrice"[^>]*>(.*?)</span>',
        html,
        re.DOTALL,
    )
    if not price_matches:
        return []
    editions = []
    for offer_index, price_html in price_matches:
        price = (
            re.sub(r'<[^>]+>', '', price_html)
            .replace('\xa0', ' ')
            .replace(' ', '.')
            .strip()
        )
        if not any(char.isdigit() for char in price):
            continue
        name = extract_offer_name(html, offer_index)
        if name is None:
            name = "Standard Edition" if len(price_matches) == 1 else f"Edition {int(offer_index) + 1}"
        editions.append({"name": name, "price_try": price})
    return editions


async def fetch_ps_store_product_page_price(product_url: str) -> dict:
    """Scrape a PS Store product page URL for all edition prices in TRY.

    Args:
        product_url: A store.playstation.com/en-tr/product/... URL.

    Returns:
        Dict with ``editions`` list and ``ps_store_product_url``, or empty dict
        if the page is an add-on or no prices are found.
    """
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(
            product_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
            },
        )
        response.raise_for_status()

    if is_ps_store_addon_page(response.text):
        return {}
    editions = scrape_ps_store_editions(response.text)
    if not editions:
        return {}
    return {"editions": editions, "ps_store_product_url": product_url}


async def find_ps_store_product_url_via_web_search(game_name: str) -> list[str]:
    """Use web search to find PS Store TR product page URLs for a game.

    Returns all matching product URLs from the search results so that callers
    can iterate and skip addon pages if the first result is not a base game.

    Args:
        game_name: Game title to search for.

    Returns:
        List of store.playstation.com/en-tr/product URLs, empty if none found.
    """
    from src.tools.web import web_search

    query = f'"{game_name}" site:store.playstation.com/en-tr/product'
    raw = await web_search.ainvoke({"query": query})
    results = json.loads(raw)

    if not isinstance(results, list):
        return []
    return [
        item.get("url", "")
        for item in results
        if "store.playstation.com/en-tr/product/" in item.get("url", "")
    ]


async def fetch_ps_store_price_via_web_search(game_name: str) -> dict:
    """Locate the PS Store TR product page via web search and scrape its prices.

    Tries each candidate URL from the web search in order, skipping any that
    are detected as Add-On pages, until a base-game page with editions is found.

    Args:
        game_name: Game title to search for.

    Returns:
        Dict with ``editions``, ``ps_store_product_url``, and ``source`` on
        success, or empty dict when no product page or prices are found.
    """
    product_urls = await find_ps_store_product_url_via_web_search(game_name)
    for product_url in product_urls:
        price_data = await fetch_ps_store_product_page_price(product_url)
        if price_data:
            return {**price_data, "source": "web_search"}
    return {}


@tool
async def get_ps_store_price_tr(game_name: str) -> str:
    """
    Get the Turkish PlayStation Store prices for a game in TRY.
    Returns all available editions (Standard, Deluxe, Ultimate, etc.) with prices.
    Finds the product page via web search, then scrapes prices directly.
    Always returns a ps_store_search_url.
    """
    store_url = f"https://store.playstation.com/en-tr/search/{quote(game_name)}"
    result: dict = {"game": game_name, "ps_store_search_url": store_url}

    try:
        price_data = await fetch_ps_store_price_via_web_search(game_name)
        if price_data:
            result.update(price_data)
        else:
            result["note"] = "Game not found in PS Store TR"
    except Exception as error:
        result["note"] = f"Price lookup failed: {error}"

    return json.dumps(result, ensure_ascii=False)


@tool
async def get_steam_player_count(game_name: str) -> str:
    """
    Get the current number of online players on Steam for a game.
    Note: PS5-exclusive games won't be on Steam.
    """
    try:
        async with httpx.AsyncClient() as client:
            found = await __find_steam_appid(game_name, client)
            if not found:
                return json.dumps({"error": f"Game '{game_name}' not found on Steam"})
            app_id, found_name = found

            players_response = await client.get(
                "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/",
                params={"appid": app_id},
            )
            players_response.raise_for_status()
            current_players = players_response.json().get("response", {}).get("player_count", 0)
            return json.dumps({
                "game": found_name,
                "appid": app_id,
                "current_players": current_players,
            }, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
async def get_steam_app_details(game_name: str) -> str:
    """
    Get Steam store details for a game: price, genres, Metacritic score,
    short description, release date, and developer.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            found = await __find_steam_appid(game_name, client)
            if not found:
                return json.dumps({"error": f"Game '{game_name}' not found on Steam"})
            app_id, _ = found

            details_response = await client.get(
                "https://store.steampowered.com/api/appdetails",
                params={"appids": app_id, "cc": "tr", "l": "en"},
            )
            details_response.raise_for_status()
            raw = details_response.json().get(str(app_id), {})
            if not raw.get("success"):
                return json.dumps({"error": "Steam API returned no data for this app"})

            data = raw["data"]
            price_overview = data.get("price_overview") or {}
            metacritic = data.get("metacritic") or {}
            return json.dumps({
                "name": data.get("name"),
                "short_description": data.get("short_description"),
                "developers": data.get("developers", []),
                "release_date": data.get("release_date", {}).get("date"),
                "genres": [genre_item["description"] for genre_item in data.get("genres", [])],
                "price_try": price_overview.get("final_formatted"),
                "metacritic_score": metacritic.get("score"),
                "steam_url": f"https://store.steampowered.com/app/{app_id}/",
            }, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
async def get_steam_reviews_summary(game_name: str) -> str:
    """
    Get the overall Steam review summary for a game: rating label,
    total reviews, and percentage of positive reviews.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            found = await __find_steam_appid(game_name, client)
            if not found:
                return json.dumps({"error": f"Game '{game_name}' not found on Steam"})
            app_id, found_name = found

            reviews_response = await client.get(
                f"https://store.steampowered.com/appreviews/{app_id}",
                params={"json": 1, "language": "all", "purchase_type": "all"},
            )
            reviews_response.raise_for_status()
            summary = reviews_response.json().get("query_summary", {})
            return json.dumps({
                "game": found_name,
                "review_score_desc": summary.get("review_score_desc"),
                "total_reviews": summary.get("total_reviews"),
                "total_positive": summary.get("total_positive"),
                "total_negative": summary.get("total_negative"),
            }, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


ALL_TOOLS = [
    get_ps_store_sales,
    get_ps_store_price_tr,
    get_steam_player_count,
    get_steam_app_details,
    get_steam_reviews_summary,
]
