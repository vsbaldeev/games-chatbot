"""
PlayStation Store and Steam tools.

Extracted from mcp_server.py and extended with:
  - get_steam_app_details — price, genres, Metacritic, description
  - get_steam_reviews_summary — overall rating and review counts

Register with: store_tools.register(mcp)
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Union
from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP

CoercedInt = Union[int, str]


async def __find_steam_appid(game_name: str, client: httpx.AsyncClient) -> tuple[int, str] | None:
    response = await client.get(
        "https://store.steampowered.com/api/storesearch/",
        params={"term": game_name, "cc": "us", "l": "en"},
    )
    response.raise_for_status()
    data = response.json()
    items = data.get("items", [])
    if not items:
        return None
    return items[0]["id"], items[0]["name"]


def register(mcp: FastMCP) -> None:
    """Register all store tools with the FastMCP server."""

    @mcp.tool()
    async def get_ps_store_sales(limit: CoercedInt = 12) -> str:
        """
        Fetch current PS Store sale titles from the psdeals.net RSS feed.
        Returns a JSON list of discounted game title strings (up to `limit` entries).
        """
        try:
            safe_limit = max(1, min(int(limit), 50))
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

    @mcp.tool()
    async def get_ps_store_price_tr(game_name: str) -> str:
        """
        Get the Turkish PlayStation Store link for a game and try to fetch its
        current price in TRY from psdeals.net. Always returns a ps_store_search_url.
        """
        store_url = f"https://store.playstation.com/tr-tr/search/{quote(game_name)}"
        result: dict = {"game": game_name, "ps_store_search_url": store_url}

        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
                response = await client.get(
                    "https://psdeals.net/tr-store",
                    params={"q": game_name},
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html,application/xhtml+xml",
                    },
                )
                response.raise_for_status()

            next_data_match = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                response.text,
                re.DOTALL,
            )
            if next_data_match:
                page_data = json.loads(next_data_match.group(1))
                games_list = (
                    page_data.get("props", {}).get("pageProps", {}).get("gamesList", [])
                )
                name_lower = game_name.lower()
                for game in games_list[:20]:
                    if name_lower not in game.get("name", "").lower():
                        continue
                    price_obj = game.get("price") or {}
                    result["name"] = game.get("name", game_name)
                    if price_obj.get("regular"):
                        result["regular_price_try"] = price_obj["regular"]
                    if price_obj.get("discount"):
                        result["sale_price_try"] = price_obj["discount"]
                    if price_obj.get("discountPercent"):
                        result["discount_percent"] = price_obj["discountPercent"]
                    psdeals_id = game.get("id") or game.get("slug") or ""
                    if psdeals_id:
                        result["psdeals_url"] = f"https://psdeals.net/tr-store/game/{psdeals_id}"
                    break
            else:
                result["note"] = "Price unavailable — use ps_store_search_url"

        except Exception as error:
            result["note"] = f"Price lookup failed: {error}"

        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
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

    @mcp.tool()
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
                    params={"appids": app_id, "l": "en"},
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
                    "genres": [g["description"] for g in data.get("genres", [])],
                    "price_usd": price_overview.get("final_formatted"),
                    "metacritic_score": metacritic.get("score"),
                    "steam_url": f"https://store.steampowered.com/app/{app_id}/",
                }, ensure_ascii=False)
        except Exception as error:
            return json.dumps({"error": str(error)})

    @mcp.tool()
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
