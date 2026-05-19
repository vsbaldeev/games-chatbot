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


def scrape_ps_store_final_price(html: str) -> str | None:
    """Extract the final purchase price from PS Store product page HTML.

    Targets the ``data-qa="mfeCtaMain#offer0#finalPrice"`` span rendered by
    the Next.js SSR layer — the only reliable source for the main game price.
    Avoids DLC and virtual-currency prices that appear elsewhere in the DOM.

    Args:
        html: Full HTML of a store.playstation.com/en-tr/product/... page.

    Returns:
        Price string (e.g. "1.399,00 TL") with non-breaking spaces normalised,
        or None if the element is absent.
    """
    match = re.search(
        r'data-qa="mfeCtaMain#offer0#finalPrice"[^>]*>(.*?)</span>',
        html,
        re.DOTALL,
    )
    if not match:
        return None
    price = re.sub(r'<[^>]+>', '', match.group(1)).replace('\xa0', ' ').replace(' ', '.').strip()
    return price if any(char.isdigit() for char in price) else None


async def fetch_ps_store_product_page_price(product_url: str) -> dict:
    """Scrape a PS Store product page URL for the current TRY price.

    Args:
        product_url: A store.playstation.com/en-tr/product/... URL.

    Returns:
        Dict with regular_price_try and ps_store_product_url, or empty dict.
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

    price = scrape_ps_store_final_price(response.text)
    if not price:
        return {}
    return {"regular_price_try": price, "ps_store_product_url": product_url}


async def find_ps_store_product_url_via_web_search(game_name: str) -> str | None:
    """Use web search to find the PS Store TR product page URL for a game.

    Args:
        game_name: Game title to search for.

    Returns:
        A store.playstation.com/en-tr/product URL, or None if not found.
    """
    from src.tools.web import web_search

    query = f'"{game_name}" site:store.playstation.com/en-tr/product'
    raw = await web_search.ainvoke({"query": query})
    results = json.loads(raw)

    if not isinstance(results, list):
        return None
    for item in results:
        url = item.get("url", "")
        if "store.playstation.com/en-tr/product/" in url:
            return url
    return None


async def fetch_ps_store_price_via_web_search(game_name: str) -> dict:
    """Locate the PS Store TR product page via web search and scrape its price.

    Uses web search restricted to store.playstation.com/en-tr/product to find
    the product URL, then fetches that page directly — never trusts aggregator
    snippet prices.

    Args:
        game_name: Game title to search for.

    Returns:
        Dict with regular_price_try, ps_store_product_url, and source on
        success, or empty dict when no product page or price is found.
    """
    product_url = await find_ps_store_product_url_via_web_search(game_name)
    if not product_url:
        return {}
    price_data = await fetch_ps_store_product_page_price(product_url)
    if not price_data:
        return {}
    return {**price_data, "source": "web_search"}


@tool
async def get_ps_store_price_tr(game_name: str) -> str:
    """
    Get the Turkish PlayStation Store price for a game in TRY.
    Finds the product page via web search, then scrapes the price directly.
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
