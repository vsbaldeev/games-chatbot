"""
IGDB / Twitch game database tools.

Extracted verbatim from mcp_server.py — no behaviour change.
Register with: igdb_tools.register(mcp)
"""

import json
import os
import re
import time
from typing import Optional, Union

import httpx
from mcp.server.fastmcp import FastMCP

CoercedInt = Union[int, str]

IGDB_API_BASE = "https://api.igdb.com/v4"
TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"

__QUERY_UNSAFE = re.compile(r'[";\\]')

__igdb_token: Optional[str] = None
__igdb_token_expiry: float = 0.0


def sanitize_igdb_string(value: str) -> str:
    """Strip characters that could break IGDB's Apicalypse query language."""
    return __QUERY_UNSAFE.sub("", value).strip()[:200]


async def get_igdb_token() -> str:
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


async def igdb_request(endpoint: str, body: str) -> list[dict]:
    token = await get_igdb_token()
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


def __register_search_games(mcp: FastMCP) -> None:
    @mcp.tool()
    async def search_games(query: str) -> str:
        """
        Search for video games by name. Returns top 5 results with id, name, and summary.
        Use this first to find a game's ID before calling get_game_details.
        """
        try:
            safe_query = sanitize_igdb_string(query)
            results = await igdb_request(
                "games",
                f'search "{safe_query}"; fields id,name,summary,first_release_date; limit 5;',
            )
            return json.dumps(results, ensure_ascii=False, indent=2)
        except httpx.HTTPStatusError as error:
            msg = f"IGDB API error (HTTP {error.response.status_code})"
            return json.dumps({"error": msg})
        except Exception as error:
            return json.dumps({"error": str(error)})


def __register_get_game_details(mcp: FastMCP) -> None:
    @mcp.tool()
    async def get_game_details(game_id: CoercedInt) -> str:
        """
        Get detailed info about a game by its IGDB ID.
        Includes platforms, genres, and multiplayer_modes.

        NOTE about crossplay: IGDB does NOT have an explicit crossplay boolean field.
        Crossplay presence can be inferred if multiplayer_modes entries exist across
        multiple platform categories. If crossplay data is ambiguous, be honest — do not guess.
        """
        try:
            safe_id = int(game_id)
            results = await igdb_request(
                "games",
                (
                    f"fields name,summary,multiplayer_modes.*,"
                    f"platforms.name,genres.name,rating,first_release_date; "
                    f"where id = {safe_id};"
                ),
            )
            return json.dumps(results, ensure_ascii=False, indent=2)
        except httpx.HTTPStatusError as error:
            msg = f"IGDB API error (HTTP {error.response.status_code})"
            return json.dumps({"error": msg})
        except Exception as error:
            return json.dumps({"error": str(error)})


def __register_find_coop_games(mcp: FastMCP) -> None:
    @mcp.tool()
    async def find_coop_games(player_count: CoercedInt, offset: CoercedInt = 0) -> str:
        """
        Find PS5 games that support online co-op for at least the given number of players.
        Returns up to 8 games sorted by rating, with multiplayer details.
        Use offset (0, 8, 16 …) to page through results.
        PS5 platform ID in IGDB is 167.
        """
        try:
            safe_count = int(player_count)
            safe_offset = max(0, min(int(offset), 64))
            results = await igdb_request(
                "games",
                (
                    f"fields name,summary,rating,multiplayer_modes.*,genres.name; "
                    f"where multiplayer_modes.onlinecoopmax >= {safe_count} & platforms = (167); "
                    f"sort rating desc; limit 8; offset {safe_offset};"
                ),
            )
            return json.dumps(results, ensure_ascii=False, indent=2)
        except httpx.HTTPStatusError as error:
            msg = f"IGDB API error (HTTP {error.response.status_code})"
            return json.dumps({"error": msg})
        except Exception as error:
            return json.dumps({"error": str(error)})


def __register_find_new_ps5_online_games(mcp: FastMCP) -> None:
    @mcp.tool()
    async def find_new_ps5_online_games(days: CoercedInt = 21) -> str:
        """
        Find PS5 games with online multiplayer released in the last N days.
        Returns up to 8 games sorted by release date descending.
        PS5 platform ID in IGDB is 167.
        """
        try:
            safe_days = max(1, min(int(days), 180))
            cutoff = int(time.time()) - safe_days * 86400
            now = int(time.time())
            results = await igdb_request(
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
            return json.dumps({"error": msg})
        except Exception as error:
            return json.dumps({"error": str(error)})


def __register_find_singleplayer_ps_games(mcp: FastMCP) -> None:
    @mcp.tool()
    async def find_singleplayer_ps_games(offset: CoercedInt = 0) -> str:
        """
        Find highly-rated single-player games available on PS5.
        Returns up to 8 games sorted by rating descending.
        Use offset (0, 8, 16 …) to get variety.
        PS5 platform ID: 167.
        """
        try:
            safe_offset = max(0, min(int(offset), 64))
            now = int(time.time())
            results = await igdb_request(
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
            msg = f"IGDB API error (HTTP {error.response.status_code})"
            return json.dumps({"error": msg})
        except Exception as error:
            return json.dumps({"error": str(error)})


def register(mcp: FastMCP) -> None:
    """Register all IGDB tools with the FastMCP server."""
    __register_search_games(mcp)
    __register_get_game_details(mcp)
    __register_find_coop_games(mcp)
    __register_find_new_ps5_online_games(mcp)
    __register_find_singleplayer_ps_games(mcp)
