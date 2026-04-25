"""
MCP server exposing game research tools.
Run as a subprocess via stdio transport: python src/mcp_server.py
IMPORTANT: Never print() to stdout — it breaks the MCP JSON-RPC channel.
           Use logging to stderr only.
"""

import json
import logging
import os
import re
import sys
from urllib.parse import quote
import time
from contextlib import asynccontextmanager
from typing import Optional, Union

CoercedInt = Union[int, str]

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

__igdb_token: Optional[str] = None
__igdb_token_expiry: float = 0.0

IGDB_API_BASE = "https://api.igdb.com/v4"
TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"

IGDB_QUERY_UNSAFE = re.compile(r'[";\\]')


def sanitize_igdb_string(value: str) -> str:
    """Strip characters that could break IGDB's Apicalypse query language."""
    return IGDB_QUERY_UNSAFE.sub("", value).strip()[:200]


@asynccontextmanager
async def lifespan(server):
    try:
        await __get_igdb_token()
        logger.info("Twitch/IGDB authentication OK")
    except KeyError as error:
        logger.error(
            f"Missing environment variable {error} — "
            "set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in your .env file"
        )
    except httpx.HTTPStatusError as error:
        logger.error(
            f"Twitch authentication failed (HTTP {error.response.status_code}: "
            f"{error.response.text}) — "
            "verify TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET at dev.twitch.tv/console"
        )
    except httpx.ConnectError as error:
        logger.error(f"Cannot reach Twitch OAuth endpoint — check network connectivity: {error}")
    except Exception as error:
        logger.error(f"Twitch startup check failed: {type(error).__name__}: {error}")
    yield


mcp = FastMCP("games-tools", lifespan=lifespan)


async def __get_igdb_token() -> str:
    global __igdb_token, __igdb_token_expiry

    if __igdb_token and time.time() < __igdb_token_expiry:
        return __igdb_token

    client_id = os.environ["TWITCH_CLIENT_ID"]
    client_secret = os.environ["TWITCH_CLIENT_SECRET"]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            TWITCH_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
        )
        response.raise_for_status()
        data = response.json()

    __igdb_token = data["access_token"]
    __igdb_token_expiry = time.time() + data["expires_in"] - 60
    return __igdb_token


async def __igdb_request(endpoint: str, body: str) -> list[dict]:
    token = await __get_igdb_token()
    client_id = os.environ["TWITCH_CLIENT_ID"]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{IGDB_API_BASE}/{endpoint}",
            headers={
                "Client-ID": client_id,
                "Authorization": f"Bearer {token}",
                "Content-Type": "text/plain",
            },
            content=body,
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def search_games(query: str) -> str:
    """
    Search for video games by name. Returns top 5 results with id, name, and summary.
    Use this first to find a game's ID before calling get_game_details.
    """
    try:
        safe_query = sanitize_igdb_string(query)
        results = await __igdb_request(
            "games",
            f'search "{safe_query}"; fields id,name,summary,first_release_date; limit 5;',
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as error:
        msg = f"IGDB API error (HTTP {error.response.status_code}) — Twitch credentials may be invalid"
        logger.error(f"search_games failed: {msg}")
        return json.dumps({"error": msg})
    except httpx.ConnectError as error:
        msg = f"Cannot reach IGDB/Twitch — check network connectivity: {error}"
        logger.error(f"search_games failed: {msg}")
        return json.dumps({"error": msg})
    except Exception as error:
        logger.error(f"search_games failed: {error}")
        return json.dumps({"error": str(error)})


@mcp.tool()
async def get_game_details(game_id: CoercedInt) -> str:
    """
    Get detailed info about a game by its IGDB ID.
    Includes platforms, genres, and multiplayer_modes.

    NOTE about crossplay: IGDB does NOT have an explicit crossplay boolean field.
    Crossplay presence can be inferred if multiplayer_modes entries exist across
    multiple platform categories (e.g. PS5 and PC both listed).
    If crossplay data is ambiguous, be honest about the limitation — do not guess.
    """
    try:
        safe_id = int(game_id)
        results = await __igdb_request(
            "games",
            (
                f"fields name,summary,multiplayer_modes.*,"
                f"platforms.name,genres.name,rating,first_release_date; "
                f"where id = {safe_id};"
            ),
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as error:
        msg = f"IGDB API error (HTTP {error.response.status_code}) — Twitch credentials may be invalid"
        logger.error(f"get_game_details failed: {msg}")
        return json.dumps({"error": msg})
    except httpx.ConnectError as error:
        msg = f"Cannot reach IGDB/Twitch — check network connectivity: {error}"
        logger.error(f"get_game_details failed: {msg}")
        return json.dumps({"error": msg})
    except Exception as error:
        logger.error(f"get_game_details failed: {error}")
        return json.dumps({"error": str(error)})


@mcp.tool()
async def get_steam_player_count(game_name: str) -> str:
    """
    Get the current number of online players on Steam for a game.
    Useful for gauging how alive a game's community is.
    Note: PS5-exclusive games won't be on Steam — the result will reflect only PC players.
    """
    try:
        async with httpx.AsyncClient() as client:
            search_response = await client.get(
                "https://store.steampowered.com/api/storesearch/",
                params={"term": game_name, "cc": "us", "l": "en"},
            )
            search_response.raise_for_status()
            search_data = search_response.json()

            items = search_data.get("items", [])
            if not items:
                return json.dumps({"error": f"Game '{game_name}' not found on Steam"})

            app_id = items[0]["id"]
            found_name = items[0]["name"]

            players_response = await client.get(
                "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/",
                params={"appid": app_id},
            )
            players_response.raise_for_status()
            players_data = players_response.json()

            current_players = players_data.get("response", {}).get("player_count", 0)
            return json.dumps({
                "game": found_name,
                "appid": app_id,
                "current_players": current_players,
            }, ensure_ascii=False)

    except Exception as error:
        logger.error(f"get_steam_player_count failed: {error}")
        return json.dumps({"error": str(error)})


@mcp.tool()
async def find_new_ps5_online_games(days: CoercedInt = 21) -> str:
    """
    Find PS5 games with online multiplayer released in the last N days.
    Returns up to 8 games sorted by release date descending, with multiplayer details.
    Use get_game_details for crossplay inference on specific results.
    PS5 platform ID in IGDB is 167.
    """
    try:
        safe_days = max(1, min(int(days), 180))
        cutoff = int(time.time()) - safe_days * 86400
        now = int(time.time())
        results = await __igdb_request(
            "games",
            (
                f"fields name,summary,rating,multiplayer_modes.*,genres.name,first_release_date; "
                f"where platforms = (167) & multiplayer_modes != null "
                f"& first_release_date > {cutoff} & first_release_date < {now}; "
                f"sort first_release_date desc; limit 8;"
            ),
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as error:
        msg = f"IGDB API error (HTTP {error.response.status_code})"
        logger.error(f"find_new_ps5_online_games failed: {msg}")
        return json.dumps({"error": msg})
    except httpx.ConnectError as error:
        msg = f"Cannot reach IGDB/Twitch — check network connectivity: {error}"
        logger.error(f"find_new_ps5_online_games failed: {msg}")
        return json.dumps({"error": msg})
    except Exception as error:
        logger.error(f"find_new_ps5_online_games failed: {error}")
        return json.dumps({"error": str(error)})


@mcp.tool()
async def get_ps_store_sales(limit: CoercedInt = 12) -> str:
    """
    Fetch current PS Store sale titles from the psdeals.net RSS feed.
    Returns a JSON list of discounted game title strings (up to `limit` entries).
    Use search_games + get_game_details to check multiplayer support for any titles
    that look interesting.
    """
    try:
        safe_limit = max(1, min(int(limit), 50))
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(
                "https://psdeals.net/rss-feed",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.text)
        titles = [
            item.text.strip()
            for item in root.findall(".//item/title")
            if item.text
        ][:safe_limit]
        return json.dumps(titles, ensure_ascii=False)
    except httpx.HTTPStatusError as error:
        msg = f"psdeals.net error (HTTP {error.response.status_code})"
        logger.error(f"get_ps_store_sales failed: {msg}")
        return json.dumps({"error": msg})
    except httpx.ConnectError as error:
        msg = f"Cannot reach psdeals.net: {error}"
        logger.error(f"get_ps_store_sales failed: {msg}")
        return json.dumps({"error": msg})
    except Exception as error:
        logger.error(f"get_ps_store_sales failed: {error}")
        return json.dumps({"error": str(error)})


@mcp.tool()
async def find_coop_games(player_count: CoercedInt, offset: CoercedInt = 0) -> str:
    """
    Find PS5 games that support online co-op for at least the given number of players.
    Returns up to 8 games sorted by rating, with multiplayer details.
    Use offset (0, 8, 16 …) to page through results and get different suggestions each call.
    PS5 platform ID in IGDB is 167.
    """
    try:
        safe_count = int(player_count)
        safe_offset = max(0, min(int(offset), 64))
        results = await __igdb_request(
            "games",
            (
                f"fields name,summary,rating,multiplayer_modes.*,genres.name; "
                f"where multiplayer_modes.onlinecoopmax >= {safe_count} & platforms = (167); "
                f"sort rating desc; limit 8; offset {safe_offset};"
            ),
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as error:
        msg = f"IGDB API error (HTTP {error.response.status_code}) — Twitch credentials may be invalid"
        logger.error(f"find_coop_games failed: {msg}")
        return json.dumps({"error": msg})
    except httpx.ConnectError as error:
        msg = f"Cannot reach IGDB/Twitch — check network connectivity: {error}"
        logger.error(f"find_coop_games failed: {msg}")
        return json.dumps({"error": msg})
    except Exception as error:
        logger.error(f"find_coop_games failed: {error}")
        return json.dumps({"error": str(error)})


@mcp.tool()
async def find_singleplayer_ps_games(offset: CoercedInt = 0) -> str:
    """
    Find highly-rated single-player games available on PS5 or PC.
    Returns up to 8 games sorted by rating descending.
    Use offset (0, 8, 16 …) to page through results and get variety.
    Only returns games with no multiplayer modes (proxy for single-player).
    PS5 platform ID: 167. PC (Windows) platform ID: 6.
    """
    try:
        safe_offset = max(0, min(int(offset), 64))
        now = int(time.time())
        results = await __igdb_request(
            "games",
            (
                f"fields name,summary,rating,genres.name,platforms.name,first_release_date; "
                f"where platforms = (167) & multiplayer_modes = null "
                f"& rating >= 75 & first_release_date < {now}; "
                f"sort rating desc; limit 8; offset {safe_offset};"
            ),
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as error:
        msg = f"IGDB API error (HTTP {error.response.status_code}) — Twitch credentials may be invalid"
        logger.error(f"find_singleplayer_ps_games failed: {msg}")
        return json.dumps({"error": msg})
    except httpx.ConnectError as error:
        msg = f"Cannot reach IGDB/Twitch — check network connectivity: {error}"
        logger.error(f"find_singleplayer_ps_games failed: {msg}")
        return json.dumps({"error": msg})
    except Exception as error:
        logger.error(f"find_singleplayer_ps_games failed: {error}")
        return json.dumps({"error": str(error)})


@mcp.tool()
async def get_ps_store_price_tr(game_name: str) -> str:
    """
    Get the Turkish PlayStation Store link for a game and try to fetch its current price
    in TRY from psdeals.net (tr-TR store). Always returns a ps_store_search_url.
    Price data is best-effort: only games currently tracked by psdeals.net will have prices.
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
                page_data
                .get("props", {})
                .get("pageProps", {})
                .get("gamesList", [])
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
        logger.warning(f"get_ps_store_price_tr price lookup failed for '{game_name}': {error}")
        result["note"] = "Price lookup failed — use ps_store_search_url"

    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
