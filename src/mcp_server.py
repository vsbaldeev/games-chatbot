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
import time
from contextlib import asynccontextmanager
from typing import Optional

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
async def get_game_details(game_id: int) -> str:
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
async def find_coop_games(player_count: int) -> str:
    """
    Find PS5 games that support online co-op for at least the given number of players.
    Returns up to 8 games sorted by rating, with multiplayer details.
    PS5 platform ID in IGDB is 167.
    """
    try:
        safe_count = int(player_count)
        results = await __igdb_request(
            "games",
            (
                f"fields name,summary,rating,multiplayer_modes.*,genres.name; "
                f"where multiplayer_modes.onlinecoopmax >= {safe_count} & platforms = (167); "
                f"sort rating desc; limit 8;"
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
