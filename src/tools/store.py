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


def __parse_ps_store_search_data(next_data: dict, game_name: str) -> dict:
    """Extract price fields from a PlayStation Store search page __NEXT_DATA__ blob.

    Args:
        next_data: Parsed JSON from the __NEXT_DATA__ script tag.
        game_name: Original search term, used as fallback name.

    Returns:
        Dict with price fields if a matching product was found, otherwise empty.
    """
    try:
        search_results = (
            next_data.get("props", {})
            .get("pageProps", {})
            .get("searchResults", {})
            .get("searchResultItems", [])
        )
        name_lower = game_name.lower()
        for item in search_results[:10]:
            item_name = item.get("name", "")
            if name_lower not in item_name.lower():
                continue
            price_obj = item.get("price") or {}
            result: dict = {"name": item_name}
            if price_obj.get("basePrice"):
                result["regular_price_try"] = price_obj["basePrice"]
            if price_obj.get("discountedPrice"):
                result["sale_price_try"] = price_obj["discountedPrice"]
            if price_obj.get("discountText"):
                result["discount_percent"] = price_obj["discountText"]
            product_id = item.get("id") or item.get("productId") or ""
            if product_id:
                result["ps_store_product_url"] = f"https://store.playstation.com/en-tr/product/{product_id}"
            return result
    except (KeyError, TypeError, AttributeError):
        pass
    return {}


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


def extract_try_price(text: str) -> str | None:
    """Extract a Turkish Lira price string from arbitrary text.

    Args:
        text: Text snippet that may contain a TRY price.

    Returns:
        The price string (e.g. "₺1.500,00") or None if not found.
    """
    for pattern in (r'₺[\d.,]+', r'[\d.,]+\s*TL\b', r'[\d.,]+\s*TRY\b'):
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            return found.group(0).strip()
    return None


async def fetch_ps_store_price_via_web_search(game_name: str) -> dict:
    """Search for Turkish PS Store price via web search.

    Used as a fallback when direct PS Store scraping fails. Queries for
    the game's price from aggregator sites (psdeals.net, psprices.com, etc.)
    and matches any TRY price pattern found in result snippets.

    Args:
        game_name: Game title to search for.

    Returns:
        Dict with name, regular_price_try, and source on success,
        or empty dict when nothing is found.
    """
    from src.tools.web import web_search

    query = f"{game_name} PlayStation Store Turkey price TL TRY fiyat"
    raw = await web_search.ainvoke({"query": query})
    results = json.loads(raw)

    if not isinstance(results, list):
        return {}

    for item in results:
        price = extract_try_price(item.get("snippet", ""))
        if not price:
            continue
        result: dict = {"regular_price_try": price, "source": "web_search"}
        title = item.get("title", "")
        if title:
            result["name"] = title
        url = item.get("url", "")
        if url:
            result["price_source_url"] = url
        return result
    return {}


async def __fetch_ps_store_price(game_name: str) -> dict:
    """Fetch Turkish PS Store price by scraping the store's own search page.

    Args:
        game_name: Game title to search for.

    Returns:
        Dict with price fields on success, or with a 'note' key on failure.
    """
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(
            f"https://store.playstation.com/en-tr/search/{quote(game_name)}",
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

    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        response.text,
        re.DOTALL,
    )
    if not match:
        return {"note": "Price unavailable — use ps_store_search_url"}
    next_data = json.loads(match.group(1))
    price_fields = __parse_ps_store_search_data(next_data, game_name)
    if not price_fields:
        return {"note": "Game not found in PS Store TR search results"}
    return price_fields


@tool
async def get_ps_store_price_tr(game_name: str) -> str:
    """
    Get the Turkish PlayStation Store price for a game in TRY.
    First scrapes the PS Store TR directly; falls back to web search
    when the page is inaccessible. Always returns a ps_store_search_url.
    """
    store_url = f"https://store.playstation.com/en-tr/search/{quote(game_name)}"
    result: dict = {"game": game_name, "ps_store_search_url": store_url}

    try:
        price_fields = await __fetch_ps_store_price(game_name)
        result.update(price_fields)
    except Exception:
        pass

    has_price = "regular_price_try" in result or "sale_price_try" in result
    if not has_price:
        try:
            web_fields = await fetch_ps_store_price_via_web_search(game_name)
            if web_fields:
                result.pop("note", None)
                result.update(web_fields)
        except Exception as error:
            result.setdefault("note", f"Price lookup failed: {error}")

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
